import keras
from keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Dropout, Dense, Bidirectional, LSTM, Add, Concatenate, Flatten, Reshape
from keras.models import Model
import os


class DeepSleepNetTA(Model):
    def __init__(self, name):
        super(DeepSleepNetTA, self).__init__(name=name)
        # cnn output 1
        self.conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(6, 1), padding="same",
                            activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3), name="TAConv1")
        self.batch_norm1 = BatchNormalization(
            momentum=0.999, epsilon=1e-5, name='teacherBatchNorm1')
        self.relu = ReLU()
        self.max_pool1 = MaxPooling2D(pool_size=(
            8, 1), strides=(8, 1), padding="same", name="TAMaxPool1")
        self.do = Dropout(0.5, name="TADropout1")
        self.conv2 = Conv2D(filters=128, kernel_size=(1, 64),
                            strides=(1, 1), padding="same", activation="relu", name="TAConv2")
        self.batch_norm2 = BatchNormalization(
            momentum=0.999, epsilon=1e-5, name='TABatchNorm2')
        self.max_pool2 = MaxPooling2D(pool_size=(
            4, 1), strides=(4, 1), padding="same", name="TAMaxPool2")
        self.reshape1 = Reshape((-1, 2048), name="TAReshape1")

        # cnn output 2
        self.conv3 = Conv2D(filters=64, kernel_size=(1, 1),
                            strides=(50, 1), padding="same", activation="relu", name="TAConv5")
        self.max_pool3 = MaxPooling2D(pool_size=(
            4, 1), strides=(4, 1), padding="same", name="TAMaxPool3")
        self.conv4 = Conv2D(filters=128, kernel_size=(1, 64),
                            strides=(1, 1), padding="same", activation="relu", name="TAConv6")
        self.max_pool4 = MaxPooling2D(pool_size=(
            2, 1), strides=(2, 1), padding="same", name="TAMaxPool4")
        self.reshape2 = Reshape((-1, 1024), name="TAReshape2")

        # combine cnn outputs
        self.concat = Concatenate(axis=-1)
        self.flatten1 = Flatten(input_shape=(1, 3072))

        # fully connected
        self.fc1 = Dense(512, activation="relu", name="TAFC1")
        self.batch_norm2 = BatchNormalization(
            momentum=0.999, epsilon=1e-5, name='TABatchNorm2')

        # rnn
        self.reshape3 = Reshape(target_shape=(-1, 3072),
                                name="TAReshape3")
        self.bidirectional = Bidirectional(
            LSTM(256), merge_mode="concat", name="TABidirectional1")

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
        output = self.relu(self.batch_norm2(self.conv2(output)))
        output = self.max_pool2(output)
        output = self.reshape1(output)
        return output

    def deep_feature_net_cnn2(self, input_var):
        output = self.relu(self.batch_norm1(self.conv3(input_var)))
        output = self.max_pool3(output)
        output = self.do(output)
        output = self.relu(self.batch_norm2(self.conv4(output)))
        output = self.max_pool4(output)
        output = self.reshape2(output)
        return output

    def deep_sleep_net_fc(self, input_layer):
        return self.relu(self.batch_norm2(self.fc1(input_layer)))

    def deep_sleep_net_rnn(self, input_layer):
        # reshape into (batch_size, seq_length, input_dim)
        output = self.reshape3(input_layer)
        output = self.bidirectional(output)
        return output


def get_deepsleepnet_TA_model(x_train, name=None, model_dir=None):
    model = DeepSleepNetTA(name=name)

    # if gave valid model directory, load weights instead of re-training
    if model_dir and os.path.isfile(model_dir):
        model.build(x_train.shape)
        model.load_weights(model_dir)

    return model
