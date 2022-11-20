import numpy as np
from skorch import NeuralNetClassifier
from torch.utils.data import random_split
import torch
import torchvision.transforms as transforms
import torchvision.datasets
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold
from classes.cnn import CNN
from classes.resnet import ResNet18
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix
import constant
from tabulate import tabulate
import argparse
import os.path


def main():
    # Default 5 epochs
    num_epochs = 5
    # Default to evaluating the model
    evaluate_model = True
    # Default to loading the model from the file
    load_from_file = True
    # Default to running K-Cross evaluation
    run_k_fold = True
    model_version = "v2"
    # Default to full and evaluating ResNet18 model
    model_save_path = constant.MODELS_DIR + model_version+"/resnet_model.pth"
    # Default to using the sample dataset
    dataset_type = "sample"
    model_type = "Resnet18"
    model = ResNet18()
    dataset_path = constant.SAMPLE_DATASET_PATH
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--load", help="Load the model from the saved file, default is true")
    arg_parser.add_argument("--epoch", help="Determine number of epochs, default is 10, max is 20")
    arg_parser.add_argument("--mode", help="Training (t) or evaluation (e), default is evaluation")
    arg_parser.add_argument("--model", help="CNN model (c) or resnet18 model (r)")
    arg_parser.add_argument("--dataset", help="Sample dataset (s) or Full dataset (f), default is sample")
    arg_parser.add_argument("--method", help="K-Fold Cross Evaluation (K) or Train/Test Split (T), default is K-Fold")

    args = arg_parser.parse_args()
    if args.__getattribute__("load") is not None:
        if args.__getattribute__("load") == '0' or args.__getattribute__("load") == '1':
            load_from_file = bool(int(args.__getattribute__("load")))
        else:
            print("Invalid selection for load param, resulting to default (load from file)")

    if args.__getattribute__("epoch") is not None:
        try:
            value = int(float(args.__getattribute__("epoch")))
            if 1 <= value <= 20:
                num_epochs = value
            else:
                print("Num epochs must be between 1 and 20, defaulting to {}".format(num_epochs))
        except:
            print("Invalid value for num epochs, defaulting to {}".format(num_epochs))

    if args.__getattribute__("mode") is not None:
        if args.__getattribute__("mode").lower() == "t":
            evaluate_model = False
        elif args.__getattribute__("mode").lower() != "e":
            print("Invalid selection for mode param, resulting to default (evaluate)")

    if args.__getattribute__("model") is not None:
        if args.__getattribute__("model").lower() == "c":
            model_save_path = constant.MODELS_DIR + model_version+"/cnn_model.pth"
            model = CNN()
            model_type = "CNN"
        elif args.__getattribute__("model").lower() != "r":
            print("Invalid selection for model param, resulting to default (Resnet18)")

    if args.__getattribute__("dataset") is not None:
        if args.__getattribute__("dataset").lower() == "f":
            dataset_path = constant.FULL_DATASET_PATH
            dataset_type = "full"
        elif args.__getattribute__("dataset").lower() != "s":
            print("Invalid selection for dataset param, resulting to default (sample dataset)")

    if args.__getattribute__("method") is not None:
        if args.__getattribute__("method").lower() == "t":
            run_k_fold = False
        elif args.__getattribute__("method").lower() != "k":
            print("Invalid selection for method param, resulting to default (K-Fold Evaluation)")

    batch_size = 10
    learning_rate = 0.001
    # Resize the images in the dataset to 32x32 to meet the CIFAR10 spec
    transform = transforms.Compose([transforms.Resize([constant.RESIZE_WIDTH, constant.RESIZE_HEIGHT]),
                                    transforms.ToTensor()])

    print("Loading {} dataset...".format(dataset_type))
    # Access the dataset from the dataset directory
    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
    # Load the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=2)
    data = next(iter(data_loader))
    # Needed to normalize the data across the colour channels
    mean = data[0].mean()
    # Needed to normalize the data across the colour channels
    std = data[0].std()
    print("...{} dataset loaded successfully!".format(dataset_type), end="\n\n")

    print("Mean of dataset = {:.4f}".format(mean))
    print("STD of dataset = {:.4f}".format(std), end="\n\n")

    # Apply the normalization transformation on the three colour channels (RGB)
    transform = transforms.Compose([transforms.Resize([constant.RESIZE_WIDTH, constant.RESIZE_HEIGHT]),
                                    transforms.ToTensor(),
                                    transforms.Normalize([mean, mean, mean], [std, std, std])])
    
    # Reload the whole dataset with the new normalization values
    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

    if load_from_file:
        print("Attempting to load {} model from file...".format(model_type))
        if os.path.exists(model_save_path):
            model.state_dict(torch.load(model_save_path))
            print("...{} model loaded from file successfully!".format(model_type), end="\n\n")
        else:
            print("Could not load {} model from file since path \"{}\" does not exist".format(model_type,
                                                                                              model_save_path),
                  end="\n\n")

    torch.manual_seed(0)
    neural_net = NeuralNetClassifier(
        model,
        max_epochs=num_epochs,
        iterator_train__num_workers=4,
        iterator_valid__num_workers=4,
        lr=learning_rate,
        batch_size=batch_size,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu")
    )

    if run_k_fold:
        perform_k_fold_cross_validation_evaluation(neural_net, dataset, evaluate_model, model_type, dataset_type, num_epochs)
    else:
        perform_train_test_split_evaluation(neural_net, dataset, evaluate_model, model_type, dataset_type, num_epochs)

    #test_for_bias(neural_net, model_type)
    print("")
    print("Saving {} model to file...".format(model_type))
    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print("...{} model saved".format(model_type))


def perform_train_test_split_evaluation(net, dataset, evaluate_model, model_type, dataset_type, num_epochs):
    print("======================================== TRAIN / TEST SPLIT EVALUATION =======================================")
    datasets = {"training": None, "testing": None}
    # Dynamically allocate 25% of the dataset to testing, and the remaining 75% to full
    datasets["training"], datasets["testing"] = train_test_split(dataset, test_size=constant.TESTING_SET_SIZE_FACTOR,

                                                                 random_state=0, shuffle=True)

    # NOTE: Need to use random_split() for net.fit() and net.predict() to work
    # Allocate 100% of the training set to training data and 0% to validation data
    training_data, _ = random_split(datasets["training"], [len(datasets["training"]), 0])
    # Allocate 100% of the testing set to testing data and 0% to validation data
    testing_data, _ = random_split(datasets["testing"], [len(datasets["testing"]), 0])
    # Fetch the full data labels
    y_train = np.array([y for x, y in iter(training_data)])

    # Train the model
    print("Training {} model on {} dataset for {} epoch(s)...".format(model_type, dataset_type, num_epochs))
    net.fit(X=training_data, y=y_train)
    print("...{} model trained".format(model_type), end="\n\n")

    if evaluate_model:
        print("Evaluating the {} model...".format(model_type))
        # Predict the results
        y_pred = net.predict(testing_data)
        print("...{} model evaluated!".format(model_type), end="\n\n")

        # Fetch the testing data labels
        y_test = np.array([y for x, y in iter(testing_data)])
        #
        binary_datasets = create_binary_datasets(y_test, y_pred, dataset.classes)

        display_measurements_per_class(binary_datasets)
        display_measurements_across_all_classes(y_test, y_pred)

        # Display which indices correspond to which labels
        for index, item in enumerate(dataset.classes):
            print("Index {} = {}".format(index, item))
        # Display the confusion matrix
        plot_confusion_matrix(net, testing_data, y_test.reshape(-1, 1))
        plt.show()


def perform_k_fold_cross_validation_evaluation(net, dataset, evaluate_model, model_type, dataset_type, num_epochs):
    print("======================================== K-FOLD CROSS-VALIDATION EVALUATION =======================================")
    num_folds = 10
    kf = KFold(shuffle=True, n_splits=num_folds)
    fold_count = 1
    fold_stats = {"Accuracy": [], "Recall": [], "Precision": [], "F1-Measure": []}
    for train_index, test_index in kf.split(dataset):
        X_train, X_test = [], []
        # Build the training set
        for index in train_index:
            X_train.append(dataset[index])
        # Build the testing set
        for index in test_index:
            X_test.append(dataset[index])

        # NOTE: Need to use random_split() for net.fit() and net.predict() to work
        # Allocate 100% of the training set to training data and 0% to validation data
        training_data, _ = random_split(X_train, [len(X_train), 0])
        # Allocate 100% of the testing set to testing data and 0% to validation data 
        testing_data, _ = random_split(X_test, [len(X_test), 0])
        y_train = np.array([y for x, y in iter(training_data)])
        # Train the model for the current fold
        print("Fold {}: Training {} model on {} dataset for {} epoch(s)...".format(fold_count, model_type, dataset_type, num_epochs))
        net.fit(X=training_data, y=y_train)
        print("...{} model trained for fold {}!".format(model_type, fold_count), end="\n\n")
        if evaluate_model:
            print("Evaluating the {} model for fold {}...".format(model_type, fold_count))
            # Predict the results
            y_pred = net.predict(testing_data)
            print("...{} model evaluated for fold {}!".format(model_type, fold_count), end="\n\n")
            # Fetch the testing data labels
            y_test = np.array([y for x, y in iter(testing_data)])
            binary_datasets = create_binary_datasets(y_test, y_pred, dataset.classes)
            display_measurements_per_class(binary_datasets, fold_count)
            display_measurements_across_all_classes(y_test, y_pred, fold_count, fold_stats)
        fold_count += 1

    agg_stats = []
    for key in fold_stats:
        sum_stat = sum(fold_stats[key])
        formatted_stat = ("{:.2f}%").format((sum_stat*100)/num_folds)
        agg_stats.append(formatted_stat)

    print(tabulate([agg_stats], headers=list(fold_stats.keys())), end="\n\n")

def test_for_bias(net, model_type):
    print("======================================== TESTING FOR BIAS =======================================")
    transform = transforms.Compose([transforms.Resize([constant.RESIZE_WIDTH, constant.RESIZE_HEIGHT]),
                                    transforms.ToTensor()])
    directories = {"age": ["adult", "child", "senior"], "gender": ["male", "female"]}
    for dir in directories:
        for sub_dir in directories[dir]:
            print("\n==========RESULTS FOR {}=========".format(sub_dir).upper())
            dataset_path = "dataset/" + dir + "/" + sub_dir
            print("Loading {} dataset...".format(sub_dir))
            # Access the bias dataset from the dataset directory
            dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
            # Load the bias dataset
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=2)
            data = next(iter(data_loader))
            # Needed to normalize the data across the colour channels
            mean = data[0].mean()
            # Needed to normalize the data across the colour channels
            std = data[0].std()
            print("...{} dataset loaded successfully!".format(sub_dir), end="\n\n")

            print("Mean of dataset = {:.4f}".format(mean))
            print("STD of dataset = {:.4f}".format(std), end="\n\n")

            # Apply the normalization transformation on the three colour channels (RGB)
            transform = transforms.Compose([transforms.Resize([constant.RESIZE_WIDTH, constant.RESIZE_HEIGHT]),
                                            transforms.ToTensor(),
                                            transforms.Normalize([mean, mean, mean], [std, std, std])])

            # Reload the whole dataset with the new normalization values
            dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=2)
            # Allocate 100% of the bias dataset to testing
            testing_data, _ = random_split(data_loader.dataset, [len(data_loader.dataset), 0])

            print("Evaluating the {} model...".format(model_type))
            # Predict the results
            y_pred = net.predict(testing_data)
            print("...{} model evaluated!".format(model_type), end="\n\n")

            # Fetch the testing data labels
            y_test = np.array([y for x, y in iter(testing_data)])
            binary_datasets = create_binary_datasets(y_test, y_pred, dataset.classes)

            display_measurements_per_class(binary_datasets)

            # Display which indices correspond to which labels
            for index, item in enumerate(dataset.classes):
                print("Index {} = {}".format(index, item))
            # Display the confusion matrix
            plot_confusion_matrix(net, testing_data, y_test.reshape(-1, 1))
            plt.show()

def create_binary_datasets(y_test, y_pred, labels):
    binary_data = {}
    for idx, label in enumerate(labels):
        binary_data[label] = {}
        binary_data[label]["y_test"] = make_dataset_binary(dataset=y_test, label=idx)
        binary_data[label]["y_pred"] = make_dataset_binary(dataset=y_pred, label=idx)
    return binary_data


def make_dataset_binary(dataset, label):
    binary_set = []
    for label_idx in dataset:
        if int(label_idx) == int(label):
            binary_set.append(1)
        else:
            binary_set.append(0)
    return binary_set


def display_measurements_per_class(binary_datasets, fold_count=-1):
    if fold_count == -1:
        print("====================Measurements per Class====================", end="\n\n")
    else:
        print("====================Measurements per Class for Fold {}====================".format(fold_count), end="\n\n")
    data = []
    calculation_types = ["Label", "Accuracy", "Recall", "Precision", "F1-Measure"]
    for label in binary_datasets:
        item = [label]
        for calc_type in calculation_types:
            if calc_type == "Accuracy":
                item.append("{:.2f}%".format(
                    accuracy_score(binary_datasets[label]["y_test"], binary_datasets[label]["y_pred"]) * 100))
            elif calc_type == "Recall":
                item.append("{:.2f}%".format(
                    recall_score(binary_datasets[label]["y_test"], binary_datasets[label]["y_pred"],
                                 average="binary", zero_division=0) * 100))
            elif calc_type == "Precision":
                item.append("{:.2f}%".format(
                    precision_score(binary_datasets[label]["y_test"], binary_datasets[label]["y_pred"],
                                    average="binary", zero_division=0) * 100))
            elif calc_type == "F1-Measure":
                item.append("{:.2f}%".format(
                    f1_score(binary_datasets[label]["y_test"], binary_datasets[label]["y_pred"],
                             average="binary", zero_division=0) * 100))
        data.append(item)
    print(tabulate(data, headers=calculation_types), end="\n\n")


def display_measurements_across_all_classes(y_test, y_pred, fold_count=-1, fold_stats=None):
    if fold_count == -1:
        print("====================Measurements Across All Classes====================")
    else:
        print("====================Measurements Across All Classes for Fold {}====================".format(fold_count))
    # Fetch the testing data labels
    calculation_types = ["Accuracy", "Recall", "Precision", "F1-Measure"]
    data = []
    item = []
    print("")
    for calc_type in calculation_types:
        score = 0
        if calc_type == "Accuracy":
            score = accuracy_score(y_test, y_pred)
            item.append("{:.2f}%".format(score * 100))
        elif calc_type == "Recall":
            score = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            item.append("{:.2f}%".format(score * 100))
        elif calc_type == "Precision":
            score = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            item.append("{:.2f}%".format(score * 100))
        elif calc_type == "F1-Measure":
            score = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            item.append("{:.2f}%".format(score * 100))

        if fold_stats is not None:
            fold_stats[calc_type].append(score)

    data.append(item)
    print(tabulate(data, headers=calculation_types), end="\n\n")

if __name__ == "__main__":
    main()
