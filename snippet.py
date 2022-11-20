binary_datasets = {}
for idx, label in enumerate(dataset.classes):
    binary_datasets[label] = {}
    binary_datasets[label]["y_test"] = make_binary(dataset=y_test, label=idx)
    binary_datasets[label]["y_pred"] = make_binary(dataset=y_pred, label=idx)

print("")



