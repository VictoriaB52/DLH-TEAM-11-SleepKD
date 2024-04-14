import keras
from keras import layers
from keras.models import Model
import os


def get_deep_sleep_teacher_model(MODEL_DIR, x_train, y_train):
    # to avoid re-training if a model is already saved, just load and return it
    if os.path.isfile(MODEL_DIR):
        return keras.models.load_model(MODEL_DIR)

    model = deep_sleep_net_teacher(x_train)

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=1, batch_size=100,
              validation_split=0.1)

    model.save(MODEL_DIR)
    return model


# build the model architecture
def deep_sleep_net_teacher(input_var):
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
    print("\nnetwork shape after pre-training " +
          str(network.shape) + '\n')

    # steps of fine-tuning model in original DSS - RNN
    fc = deep_sleep_net_fc(network)
    rnn = deep_sleep_net_rnn(network)
    final_output = layers.Add()([fc, rnn])
    final_output = layers.Dropout(0.5)(final_output)

    # soft labels of 5 possible sleep stages
    final_output = layers.Dense(5, activation="softmax")(final_output)
    print("\nnetwork shape after fine-tuning " +
          str(final_output.shape) + '\n')

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
    # reshape into (batch_size, seq_length, input_dim)
    output = layers.Reshape(input_shape=input_layer.shape,
                            target_shape=(-1, 3072), name="teacherReshape3")(input_layer)
    output = layers.Bidirectional(
        layers.LSTM(512), merge_mode="concat", name="teacherBidirectional1")(output)
    return output
