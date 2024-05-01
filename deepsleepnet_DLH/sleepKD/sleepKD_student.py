import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Reshape, Bidirectional, LSTM
from deepsleepnet_DLH.sleepKD.sleepKD_base import DeepSleepNetBase, DeepSleepNetModelBase
from deepsleepnet_DLH.sleepKD.sleepKD import SleepKD


class DeepSleepNetStudent(DeepSleepNetBase):
    def __init__(self, *args, **kwargs):
        super(DeepSleepNetStudent, self).__init__(*args, **kwargs)

        self.model: DeepSleepNetStudentModel | None = None

    def get_model(self, is_student: bool = True,):

        self.model = DeepSleepNetStudentModel(
            self.name, self.teacher_model)
        self.model.is_student = is_student
        self.train()

        return self.model


class DeepSleepNetStudentModel(DeepSleepNetModelBase):
    def __init__(self, *args, **kwargs):
        super(DeepSleepNetStudentModel, self).__init__(*args, **kwargs)

        # cnn output 1
        self.conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(6, 1), padding="same",
                            activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3), name="StudentConv1")
        self.max_pool1 = MaxPooling2D(pool_size=(
            8, 1), strides=(8, 1), padding="same", name="StudentMaxPool1")

        # cnn output 2
        self.conv2 = Conv2D(filters=64, kernel_size=(1, 1),
                            strides=(50, 1), padding="same", activation="relu", name="StudentConv5")
        self.max_pool2 = MaxPooling2D(pool_size=(
            4, 1), strides=(4, 1), padding="same", name="StudentMaxPool3")

        # fully connected
        self.fc1 = Dense(1024, activation="relu", name="StudentFC1")

        # rnn
        self.reshape1 = Reshape(
            target_shape=(-1, 3072), name="StudentFineTuneReshape1")
        self.bidirectional = Bidirectional(
            LSTM(128), merge_mode="concat", name="StudentBidirectional1")

        # to classes
        self.fc2 = Dense(5, activation="softmax", name="StudentFC2")

        # sleepKD loss layer
        self.sleepKD = SleepKD()

    def deep_feature_net_cnn1(self, input_layer):
        output = self.conv1(input_layer)
        output = self.max_pool1(output)
        output = self.do(output)
        output = self.flatten(output)
        return output

    def deep_feature_net_cnn2(self, input_var):
        output = self.conv2(input_var)
        output = self.max_pool2(output)
        output = self.do(output)
        output = self.flatten(output)
        return output

    def deep_sleep_net_fc(self, input_layer):
        return self.fc1(input_layer)

    def deep_sleep_net_rnn(self, input_layer):
        # reshape into (batch_size, seq_length, input_dim)
        output = self.reshape1(input_layer)
        output = self.bidirectional(output)
        return output

    def deep_sleep_net_final_output(self, input):
        return self.fc2(input)
