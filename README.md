# COMP 472 Image Recognition Project

The goal of this project is to teach a Convolutional Neural Network to classify what type of mask a person is wearing, 
or if they're wearing any at all, given a set of images. Our dataset consists of around 1600 images equally balanced across the
four classes. Of the 1600 images, about 75% of them are allocated both randomly and dynamically to train the CNN, whereas the 
remaining 25% are used to evaluate the CNN. The four classes of masks that we have are: 

* none (i.e no mask)
* cloth
* surgical
* n95

## Submitted Files

* ### main.py
  * Contains the logic for loading the dataset, initializing, training, and testing the CNN, and displaying the results.
* ### CNN.py
  * Contains the Convolutional Network class that we train and evaluate in **main.py**
* ###constant.py
  * Defines constants used within **main.py**
* ###saved_models/model
  * Contains the state dictionary of the CNN model that we save to and load from in  **main.py**
* ###dataset/training
  * The full dataset containing all 1600 images, evenly balanced between each class
* ###dataset/sample
  * The sample dataset containing 100 images, evenly balanced between each class
  
## How to Run the Project
Before starting, we must create a valid Anaconda environment running Python versions 3.7-3.9.
Once the environment has been created, install the required libraries by running these 4 commands in the Anaconda terminal within your environment:

```pycon
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c conda-forge skorch
conda install scikit-learn
conda install matplotlib
```
Once that's done, link the project to your Anaconda environment from within your IDE if you want to be able to run the project from your IDE.

To run the project from the terminal on Windows, open the Anaconda terminal, activate the environment you just created, and go to the project directory. 

To run the project from the terminal without passing any parameters, run:
```pycon
python main.py
```
To run the project from the terminal with passing parameters, run:
```pycon
python main.py --epoch val --load val2 --dataset val3 --mode val4 --method val5
```
**Note that all parameters are optional and any combination of them can be passed**

Parameter definitions:
* **--epoch** : Sets the number of epochs we train the model for. 
  * Default = 10, 
  * Minimum = 1 
  * Maximum = 20
* **--load** : Determines whether to load the model from the saved file 
  * Default = 1 (True, load from the file)
  * Minimum = 0 (False, don't load from the file)
  * Maximum = 1 (True, load from the file)
* **-dataset**: Determines whether to train and evaluate the model using the sample dataset (100 images) or the full dataset (~1600 images)
  * Default = **s** (use sample dataset)
  * Valid values:
    * **s** (use sample dataset)
    * **f** (use full dataset)
* **--mode**: Determines whether we only train the model, or if we train the model AND evaluate it
  * Default = **e** (train and evaluate the model)
  * Valid values:
  * **e** (train and evaluate the model)
  * **t** (ONLY train the model)
* **--method**: Select which test and evaluation method to use
  * Default = **K** (K-Fold Cross-Evaluation)
  *  Valid values:
    * **K** (K-Fold Cross-Evaluation)
    * **T** (Train/Test Split)