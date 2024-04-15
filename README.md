# Deep Learning For Healthcare Final Project: Team 111

## Members: Victoria Buszek (vbuszek2@illinois.edu)

## Original Paper: Teacher Assistant-Based Knowledge Distillation Extracting Multi-level Features on Single Channel Sleep EEG

Link to Original Repo: https://github.com/HychaoWang/SleepKD

The SleepKD knowledge distillation method was tested on two models: SalientSleepNet and DeepSleepNet. I was only able to do work
on DeepSleepNet so far. All of the files in the DeepSleepNet Files are created by the authors of DeepSleepNet: a Model for Automatic Sleep Stage
Scoring based on Raw Single-Channel EEG. The source code can be found at
https://github.com/akaraspt/deepsleepnet/tree/master?tab=readme-ov-file Additionally, the data folder containts the results
of running the DeepSleepNet file prepare_phsionet.py. It is contained in the main folder for ease of access, but it is also
not a result of any preprocessing I wrote myself.

To run the project locally, the following libraries are needed

- Tensorflow 2.15.1
- Keras 2.15.0
- Numpy 1.26.4
- mne 1.6.1
- sklearn 1.2.1

Using Python 3.11 interpretor in VSCode

Simply clone the repository and run `python main.py` in the root. The preprocessed dataset in the `data` folder is ready for use, so we just load the .npz files and split our data into train/test set. The file will go through creating/training a new teacher DeepSleepNet model if one does not exist at the directory `deep_sleep_net_teacher.keras`(it will not be when first creating the repo because the file is large, so I've included it in my .gitignore config). Once trained, the teacher model will be saved and will do predictions on the test set.

After this, we make a DeepSleepNet student model if one does not exist at `deep_sleep_net_student.keras ` and follow up with a Distiller model. We distill the teacher information to the student and train the mode on the same training data used for the techer, again performing some predictions.
