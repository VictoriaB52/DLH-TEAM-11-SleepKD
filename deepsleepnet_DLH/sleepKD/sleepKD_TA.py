import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Reshape, Bidirectional, LSTM
from deepsleepnet_DLH.sleepKD.sleepKD_base import DeepSleepNetBase, DeepSleepNetModelBase


class DeepSleepNetTA(DeepSleepNetBase):
    def __init__(self, name: str = None):
        super(DeepSleepNetTA, self).__init__(name=name)

        self.name = name
        self.model: DeepSleepNetTAModel | None = None

    def get_model(self, should_train: bool = False):
        self.model = DeepSleepNetTAModel(self.name)

        if should_train:
            self.train()

        return self.model


class DeepSleepNetTAModel(DeepSleepNetModelBase):
    def __init__(self, name):
        super(DeepSleepNetTAModel, self).__init__(name=name)

     # cnn output 1
        self.conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(6, 1), padding="same",
                            activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3), name="TAConv1")
        self.max_pool1 = MaxPooling2D(pool_size=(
            8, 1), strides=(8, 1), padding="same", name="TAMaxPool1")

        self.do = Dropout(0.5, name="TADropout1")
        self.conv2 = Conv2D(filters=128, kernel_size=(1, 64),
                            strides=(1, 1), padding="same", activation="relu", name="TAConv2")
        self.max_pool2 = MaxPooling2D(pool_size=(
            4, 1), strides=(4, 1), padding="same", name="TAMaxPool2")

        # cnn output 2
        self.conv3 = Conv2D(filters=64, kernel_size=(1, 1),
                            strides=(50, 1), padding="same", activation="relu", name="TAConv5")
        self.max_pool3 = MaxPooling2D(pool_size=(
            4, 1), strides=(4, 1), padding="same", name="TAMaxPool3")
        self.conv4 = Conv2D(filters=128, kernel_size=(1, 64),
                            strides=(1, 1), padding="same", activation="relu", name="TAConv6")
        self.max_pool4 = MaxPooling2D(pool_size=(
            2, 1), strides=(2, 1), padding="same", name="TAMaxPool4")

        # fully connected
        self.fc1 = Dense(512, activation="relu", name="TAFC1")

        # rnn
        self.reshape1 = Reshape(
            target_shape=(-1, 3072), name="TAFineTuneReshape1")
        self.bidirectional = Bidirectional(
            LSTM(512), merge_mode="concat", name="TABidirectional1")

        # to classes
        self.fc2 = Dense(5, activation="softmax")

    def deep_feature_net_cnn1(self, input):
        output = self.conv1(input)
        output = self.max_pool1(output)
        output = self.do(output)
        output = self.conv2(output)
        output = self.max_pool2(output)
        output = self.flatten(output)
        return output

    def deep_feature_net_cnn2(self, input):
        output = self.conv3(input)
        output = self.max_pool3(output)
        output = self.do(output)
        output = self.conv4(output)
        output = self.max_pool4(output)
        output = self.flatten(output)
        return output

    def deep_sleep_net_fc(self, input):
        return self.fc1(input)

    def deep_sleep_net_rnn(self, input):
        # reshape into (batch_size, seq_length, input_dim)
        output = self.flatten(input)
        output = self.bidirectional(output)
        return output

    def deep_sleep_net_final_output(self, input):
        return self.fc2(input)
