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

All the files I created for the project are in `./deepsleepnet_DLH/`. This directory is itself split into two parts: `/pretrain_finetune/` and `/sleepKD/`. `/pretrain_finetune/` contains work for replicating the DeepSleepNet model to the best of my ability. In this directory, `data_loader.py` file with logic for loading and transforming data to prepare it as a model input. `deepsleepnet_base.by` contains the common logic for creating a model and passing it through pre-train and fine-tune layers. The specific teacher, teacher assistant, and student architectures are defined in their own files here. The distillers used to distill knowledge into the pretrain and fine-tune models respectively are all in `deepsleepnet_distiller`. Finally, `deepsleepnet_main.py` runs the logic for creating and evaluating each of the models.

The other repo in the folder, `sleepKD`, contains the work I was getting started on regarding incorporating the SleepKD loss layer from my project's paper into the DeepSleepNet architecture. Similar to `pretrain_finetune`, there were three architectures in addition to the distiller defined, though here the pretrain and finetune layers were combined into one model. If I'd gotten SleepKD working, the distiller would have served as a comparison to the SleepKD method. The base for the models in this repo also includes a custom training loop that passes the parameters needed for the SleepKD layer to calculate losses when training.

To run the project, simply clone the repository and run `python main.py` in the root. If everything had gotten finished, this file woulf be calling functions to run the pretrain/finetune and sleepKD main files. However, since only pretrain/finetune have a working version, the `# run_sleepkd_deepsleepnet()` function has been commented out.

The preprocessed dataset in the `data` folder is ready for use, so we just load the .npz files and split our data into train/test set. The file will go through creating/training a new teacher DeepSleepNet model if weights have not been saved at the directory specified. In our case, the weights are saved in `deepsleepnet_DLH/pretrain_finetune/models/`. The weights will not be saved first creating the repo because the files iare large, so I've included .h5 files in my .gitignore config. Once trained, the models' weights will be saved and the model will be evaluated.
