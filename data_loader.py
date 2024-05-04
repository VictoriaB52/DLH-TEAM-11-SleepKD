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


# CREDIT: This file is heavily inspired by DeepSleepNet's SeqDataLoader._load_npz_list_files function
# https://github.com/akaraspt/deepsleepnet/blob/master/deepsleep/data_loader.py
#
# Given a list of npz_files, load each and save features and labels into lists
def _load_npz_list_files(npz_files):
    """Load data and labels from list of npz files."""
    data = []
    labels = []
    fs = None
    for npz_f in npz_files:
        # print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate = _load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")

        tmp_data = np.squeeze(tmp_data)
        tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

        data.append(tmp_data)
        labels.append(tmp_labels)

    return data, labels


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

    # returning list of data, labels for each file
    data, labels = _load_npz_list_files(
        npz_files=npzfiles)

    return data, labels


# stack data from lists of features and labels to get input for pre-training
# this logic is originally in load_npz_list_files in DeepSleepNet for NonSeqDataLoader
def process_non_seq(data, labels):

    data = np.vstack(data)
    labels = np.hstack(labels)

    # Casting
    data = data.astype(np.float32)
    labels = labels.astype(np.int32)

    # Use balanced-class, oversample training set
    data, labels = get_balance_class_oversample(
        x=data, y=labels
    )

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


# CREDIT: this file is taken directly from DeepSleepNet's utils.py
# https://github.com/akaraspt/deepsleepnet/blob/master/deepsleep/utils.py
def iterate_batch_seq_minibatches(inputs, targets, batch_size, seq_length):
    assert len(inputs) == len(targets)
    n_inputs = len(inputs)
    batch_len = n_inputs // batch_size
    epoch_size = batch_len // seq_length

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or seq_length")

    seq_inputs = np.zeros((batch_size, batch_len) + inputs.shape[1:],
                          dtype=inputs.dtype)
    seq_targets = np.zeros((batch_size, batch_len) + targets.shape[1:],
                           dtype=targets.dtype)

    for i in range(batch_size):
        seq_inputs[i] = inputs[i*batch_len:(i+1)*batch_len]
        seq_targets[i] = targets[i*batch_len:(i+1)*batch_len]

    for i in range(epoch_size):
        x = seq_inputs[:, i*seq_length:(i+1)*seq_length]
        y = seq_targets[:, i*seq_length:(i+1)*seq_length]
        flatten_x = x.reshape((-1,) + inputs.shape[1:])
        flatten_y = y.reshape((-1,) + targets.shape[1:])
        # returning reshaped values to work as inputs to pre-train portion of model
        yield flatten_x, flatten_y

