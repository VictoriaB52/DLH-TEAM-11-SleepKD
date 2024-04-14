import os
import numpy as np


# CREDIT: This file is taken directly from DeepSleepNet's data_loader.py
# https://github.com/akaraspt/deepsleepnet/blob/master/deepsleep/data_loader.py
#
# Extracts features, labels, and sampling rate for a single npz_file
def _load_npz_file(npz_file):
    """Load data and labels from a npz file."""
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    return data, labels, sampling_rate


# CREDIT: This file is taken directly from DeepSleepNet's data_loader.py
# https://github.com/akaraspt/deepsleepnet/blob/master/deepsleep/data_loader.py
#
# Given a list of npz_files, process each and store features/labels into their own file
def _load_npz_list_files(npz_files):
    """Load data and labels from list of npz files."""
    data = []
    labels = []
    fs = None
    print(len(npz_files))
    print(npz_files)
    for npz_f in npz_files:
        print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate = _load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")
        data.append(tmp_data)
        labels.append(tmp_labels)
    data = np.vstack(data)
    labels = np.hstack(labels)
    return data, labels


# CREDIT: this file is taken directly from DeepSleepNet's utils.py
# https://github.com/akaraspt/deepsleepnet/blob/master/deepsleep/utils.py
#
# To prevent a single sleep stage from being overrepresented in the dataset and
# skewring results, oversample from the given files and get a more even distribution
def get_balance_class_oversample(x, y):
    """
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y


# CREDIT: this function is heavily inspired by the load_train_data() function in DeepSleepNet's data_loader.py
# https://github.com/akaraspt/deepsleepnet/blob/master/deepsleep/data_loader.py
#
# Since we handle splitting data using sklearn (for train/test) and the validation_split parameter
# of keras.models.Model.fit(), we don't need to worry about fold index. We just load features/labels
# for all npz files
def load_all_data(data_dir):
    allfiles = os.listdir(data_dir)
    npzfiles = []
    for idx, f in enumerate(allfiles):
        if ".npz" in f:
            npzfiles.append(os.path.join(data_dir, f))

    data, labels = _load_npz_list_files(
        npz_files=npzfiles)

    # Reshape the data to match the input of the model - conv2d
    data = np.squeeze(data)
    data = data[:, :, np.newaxis, np.newaxis]

    # Casting
    data = data.astype(np.float32)
    labels = labels.astype(np.int32)

    # Use balanced-class, oversample training set
    data, labels = get_balance_class_oversample(
        x=data, y=labels
    )

    return data, labels
