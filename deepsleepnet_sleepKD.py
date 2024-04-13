from deepsleep.data_loader import NonSeqDataLoader
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import os

# only time we're directly using code from a DeepSleepNet module
# added load_all_data() method to NonSeqDataLoader
import sys
sys.path.append('./deepsleepnet')


def get_deep_sleep_model(MODEL_DIR, x_train, y_train):
    if os.path.isfile(MODEL_DIR):
        return keras.models.load_model(MODEL_DIR)

    model = deep_sleep_net(x_train)

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=1, batch_size=100,
              validation_split=0.1)

    model.save(MODEL_DIR)
    return model


def deep_sleep_net(input_var):
    wd = 1e-3
    input_layer = layers.Input(
        shape=(input_var.shape[1], input_var.shape[2], input_var.shape[3]))

    cnn1 = deep_feature_net_cnn1(input_layer, wd)
    cnn2 = deep_feature_net_cnn2(input_layer, wd)
    network = layers.Concatenate(axis=-1)([cnn1, cnn2])
    network = layers.Dropout(0.5)(network)
    print("network shape before reshape for deepfeaturenet " + str(network.shape))

    # reshape and softmax layers of pretrain model in original DSS
    network = layers.Flatten(input_shape=(1, 3072))(network)
    print("network shape after supposed end of pre train " + str(network.shape))
    # final_output = layers.Dense(5, activation="softmax")(final_output)

    fc = deep_sleep_net_fc(network)
    rnn = deep_sleep_net_rnn(network)
    final_output = layers.Add()([fc, rnn])
    final_output = layers.Dropout(0.5)(final_output)
    final_output = layers.Dense(5, activation="softmax")(final_output)
    model = Model(inputs=input_layer, outputs=final_output)
    return model


def deep_feature_net_cnn1(input_layer, wd):

    output = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(6, 1), padding="same",
                           activation="relu", kernel_regularizer=keras.regularizers.l2(wd), name="teacherConv1")(input_layer)

    output = layers.MaxPooling2D(pool_size=(
        8, 1), strides=(8, 1), padding="same", name="teacherMaxPool1")(output)

    output = layers.Dropout(0.5, name="teacherDropout1")(output)

    output = layers.Conv2D(filters=128, kernel_size=(1, 64),
                           strides=(1, 1), padding="same", activation="relu", name="teacherConv2")(output)
    output = layers.Conv2D(filters=128, kernel_size=(1, 128),
                           strides=(1, 1), padding="same", activation="relu", name="teacherConv3")(output)
    output = layers.Conv2D(filters=128, kernel_size=(1, 128),
                           strides=(1, 1), padding="same", activation="relu", name="teacherConv4")(output)

    output = layers.MaxPooling2D(pool_size=(
        4, 1), strides=(4, 1), padding="same", name="teacherMaxPool2")(output)
    output = layers.Reshape((-1, 2048), name="teacherReshape1")(output)
    return output


def deep_feature_net_cnn2(input_var, wd):
    output = layers.Conv2D(filters=64, kernel_size=(1, 1),
                           strides=(50, 1), padding="same", activation="relu", name="teacherConv5")(input_var)

    output = layers.MaxPooling2D(pool_size=(
        4, 1), strides=(4, 1), padding="same", name="teacherMaxPool3")(output)

    output = layers.Dropout(0.5, name="teacherDropout2")(output)

    output = layers.Conv2D(filters=128, kernel_size=(1, 64),
                           strides=(1, 1), padding="same", activation="relu", name="teacherConv6")(output)
    output = layers.Conv2D(filters=128, kernel_size=(1, 128),
                           strides=(1, 1), padding="same", activation="relu", name="teacherConv7")(output)
    output = layers.Conv2D(filters=128, kernel_size=(1, 128),
                           strides=(1, 1), padding="same", activation="relu", name="teacherConv8")(output)

    output = layers.MaxPooling2D(pool_size=(
        2, 1), strides=(2, 1), padding="same", name="teacherMaxPool4")(output)
    output = layers.Reshape((-1, 1024), name="teacherReshape2")(output)
    return output


def deep_sleep_net_fc(input_layer):
    print("fc on input layer " + str(input_layer.shape))
    return layers.Dense(1024, activation="relu", name="teacherFC1")(input_layer)


def deep_sleep_net_rnn(input_layer):
    # initial_state_fw = tf.zeros((input_layer.shape[0]))
    # reshape into (batch_size, seq_length, input_dim)

    print("trying to reshape " + str(input_layer.shape))
    output = layers.Reshape(input_shape=input_layer.shape,
                            target_shape=(-1, 3072), name="teacherReshape3")(input_layer)
    output = layers.Bidirectional(
        layers.LSTM(512), merge_mode="concat", name="teacherBidirectional1")(output)
    return output


def main():
    DATA_DIR = 'data/eeg_fpz_cz/'
    MODEL_DIR = 'deepsleepnetSCCE.keras'
    n_folds = 20
    fold_idx = 0
    EPOCH_SEC_LEN = 30

    np.random.seed(12345)
    tf.random.set_seed(12345)

    data_loader = NonSeqDataLoader(
        data_dir=DATA_DIR,
        n_folds=n_folds,
        fold_idx=fold_idx)
    data, labels = data_loader.load_all_data()

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.1, random_state=12345)

    deep_sleep_net_model = get_deep_sleep_model(MODEL_DIR, x_train, y_train)
    preds = deep_sleep_net_model.predict(x_test, batch_size=100)

    print(labels)
    preds = np.argmax(preds, axis=1)
    print(preds)

    conf_matrix = confusion_matrix(y_test, preds)
    print(conf_matrix)
    print(accuracy_score(y_test, preds))
    print(f1_score(y_test, preds, average="macro"))


main()


#  FINE TUNED MODEL ONLY CODE - KEEP JUST IN CASE

# def get_finetuned_model(FINE_TUNED_MODEL_DIR, PRE_TRAINED_MODEL_DIR, DATA_DIR, n_folds=20, fold_idx=0):

#     if os.path.isfile(FINE_TUNED_MODEL_DIR):
#         return keras.models.load_model(FINE_TUNED_MODEL_DIR)

#     pretrained_model = get_pretrained_model(
#         PRE_TRAINED_MODEL_DIR, DATA_DIR, n_folds, fold_idx)

#     data_loader = SeqDataLoader(
#         data_dir=DATA_DIR,
#         n_folds=n_folds,
#         fold_idx=fold_idx)

#     x_train, y_train, x_valid, y_valid = data_loader.load_train_data()

#     model = deep_sleep_net_model(
#         pretrained_model.input, pretrained_model.output)

#     model.compile(optimizer='adam',
#                   loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

#     model.fit(x_train, y_train, epochs=1, batch_size=100,
#               validation_data=(x_valid, y_valid))

#     model.save(FINE_TUNED_MODEL_DIR)
#     return model
