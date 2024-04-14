import keras
from keras import layers
from keras.models import Model
import os


def get_deep_sleep_student_model(MODEL_DIR, x_train):
    # to avoid re-training if a model is already saved, just load and return it
    if os.path.isfile(MODEL_DIR):
        return keras.models.load_model(MODEL_DIR)

    model = deep_sleep_net_student(x_train)

    # don't actually fit the student model - done in distiller in main file
    return model


# build the model architecture
def deep_sleep_net_student(input_var):
    wd = 1e-3
    input_layer = layers.Input(
        shape=(input_var.shape[1], input_var.shape[2], input_var.shape[3]))

    # steps of pre-training model in original DSS - Convolution
    cnn1 = deep_feature_net_cnn1(input_layer, wd)
    cnn2 = deep_feature_net_cnn2(input_layer, wd)
    network = layers.Concatenate(axis=-1)([cnn1, cnn2])
    network = layers.Dropout(0.5)(network)

    # final layer of pretrain model in original DSS
    network = layers.Flatten(input_shape=(1, 3072))(network)
    print("\nnetwork shape after end of pre-train " +
          str(network.shape) + '\n')

    # steps of fine-tuning model in original DSS - RNN
    fc = deep_sleep_net_fc(network)
    rnn = deep_sleep_net_rnn(network)
    final_output = layers.Add()([fc, rnn])
    final_output = layers.Dropout(0.5)(final_output)

    # soft labels of 5 possible sleep stages
    final_output = layers.Dense(5, activation="softmax")(final_output)
    model = Model(inputs=input_layer, outputs=final_output)
    return model


def deep_feature_net_cnn1(input_layer, wd):

    output = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(6, 1), padding="same",
                           activation="relu", kernel_regularizer=keras.regularizers.l2(wd), name="studentConv1")(input_layer)

    output = layers.MaxPooling2D(pool_size=(
        8, 1), strides=(8, 1), padding="same", name="studentMaxPool1")(output)

    output = layers.Dropout(0.5, name="studentDropout1")(output)

    output = layers.Conv2D(filters=128, kernel_size=(1, 64),
                           strides=(1, 1), padding="same", activation="relu", name="studentConv2")(output)

    output = layers.MaxPooling2D(pool_size=(
        4, 1), strides=(4, 1), padding="same", name="studentMaxPool2")(output)
    output = layers.Reshape((-1, 2048), name="studentReshape1")(output)
    return output


def deep_feature_net_cnn2(input_var, wd):
    output = layers.Conv2D(filters=64, kernel_size=(1, 1),
                           strides=(50, 1), padding="same", activation="relu", name="studentConv3")(input_var)

    output = layers.MaxPooling2D(pool_size=(
        4, 1), strides=(4, 1), padding="same", name="studentMaxPool3")(output)

    output = layers.Dropout(0.5, name="studentDropout2")(output)

    output = layers.Conv2D(filters=128, kernel_size=(1, 64),
                           strides=(1, 1), padding="same", activation="relu", name="studentConv4")(output)

    output = layers.MaxPooling2D(pool_size=(
        2, 1), strides=(2, 1), padding="same", name="studentMaxPool4")(output)
    output = layers.Reshape((-1, 1024), name="studentReshape2")(output)
    return output


def deep_sleep_net_fc(input_layer):
    print("fc on input layer " + str(input_layer.shape))
    return layers.Dense(512, activation="relu", name="studentFC1")(input_layer)


def deep_sleep_net_rnn(input_layer):
    # reshape into (batch_size, seq_length, input_dim)
    output = layers.Reshape(input_shape=input_layer.shape,
                            target_shape=(-1, 3072), name="studentReshape3")(input_layer)
    output = layers.Bidirectional(
        layers.LSTM(256), merge_mode="concat", name="studentBidirectional1")(output)
    return output
