import keras
from keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Dropout, Dense, Bidirectional, LSTM, Add, Concatenate, Flatten, Reshape
from keras.models import Model
import os


class DeepSleepNetStudent(Model):
    def __init__(self, name):
        super(DeepSleepNetStudent, self).__init__(name=name)
        # cnn output 1
        self.conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(6, 1), padding="same",
                            activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3), name="studentConv1")
        self.batch_norm1 = BatchNormalization(
            momentum=0.999, epsilon=1e-5, name="studentBatchNorm1")
        self.relu = ReLU()
        self.max_pool1 = MaxPooling2D(pool_size=(
            8, 1), strides=(8, 1), padding="same", name="studentMaxPool1")
        self.do = Dropout(0.5, name="studentDropout1")

        self.reshape1 = Reshape((-1, 4032), name="studentReshape1")

        # cnn output 2
        self.conv2 = Conv2D(filters=64, kernel_size=(1, 1),
                            strides=(50, 1), padding="same", activation="relu", name="studentConv2")
        self.max_pool2 = MaxPooling2D(pool_size=(
            4, 1), strides=(4, 1), padding="same", name="studentMaxPool2")
        self.reshape2 = Reshape((-1, 960), name="studentReshape2")

        # combine cnn outputs
        self.concat = Concatenate(axis=-1)
        self.flatten1 = Flatten(input_shape=(1, 4992))

        # fully connected
        self.fc1 = Dense(256, activation="relu", name="studentFC1")
        self.batch_norm2 = BatchNormalization(
            momentum=0.999, epsilon=1e-5, name='teacherBatchNorm2')

        # rnn
        self.reshape3 = Reshape(target_shape=(-1, 4992),
                                name="studentReshape3")
        self.bidirectional = Bidirectional(
            LSTM(128), merge_mode="concat", name="studentBidirectional1")

        # combine fc + rnn
        self.add = Add()
        # to classes
        self.fc2 = Dense(5, activation="softmax")

    def call(self, input):
        # steps of pre-training model in original DSS - Convolution
        cnn1 = self.deep_feature_net_cnn1(input)
        cnn2 = self.deep_feature_net_cnn2(input)
        network = self.concat([cnn1, cnn2])
        network = self.do(network)

        # final layer of pretrain model in original DSS
        network = self.flatten1(network)
        print("\nnetwork shape after pre-training: {}".format(network.shape))

        # steps of fine-tuning model in original DSS - RNN
        fc = self.deep_sleep_net_fc(network)
        rnn = self.deep_sleep_net_rnn(network)
        final_output = self.add([fc, rnn])
        final_output = self.do(final_output)

        # soft labels of 5 possible sleep stages
        final_output = self.fc2(final_output)
        print("network shape after fine-tuning: {}".format(final_output.shape))

        return final_output

    def deep_feature_net_cnn1(self, input_layer):

        output = self.relu(self.batch_norm1(self.conv1(input_layer)))
        output = self.max_pool1(output)
        output = self.do(output)
        output = self.reshape1(output)
        return output

    def deep_feature_net_cnn2(self, input_var):
        output = self.relu(self.batch_norm1(self.conv2(input_var)))
        output = self.max_pool2(output)
        output = self.do(output)
        output = self.reshape2(output)
        return output

    def deep_sleep_net_fc(self, input_layer):
        return self.relu(self.batch_norm2(self.fc1(input_layer)))

    def deep_sleep_net_rnn(self, input_layer):
        # reshape into (batch_size, seq_length, input_dim)
        output = self.reshape3(input_layer)
        output = self.bidirectional(output)
        return output


def get_deepsleepnet_student_model(x_train, y_train=None, name=None, return_trained=True, training_epochs=1, model_dir=None):

    model = DeepSleepNetStudent(name=name)
    if not return_trained:
        return model

    # if gave valid model directory, load weights instead of re-training
    if model_dir and os.path.isfile(model_dir):
        model.build(x_train.shape)
        model.load_weights(model_dir)
        return model

    if y_train is None:
        raise Exception(
            "Could not load model and no y_train was provided - can not train new model")

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=training_epochs, batch_size=100,
              validation_split=0.1)

    model.save_weights(
        model_dir if model_dir else 'models/deepsleepnet_student_weights.h5')
    return model
