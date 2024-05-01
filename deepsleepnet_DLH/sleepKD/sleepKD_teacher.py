import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Reshape, Bidirectional, LSTM
from deepsleepnet_DLH.sleepKD.sleepKD_base import DeepSleepNetBase, DeepSleepNetModelBase


class DeepSleepNetTeacher(DeepSleepNetBase):
    def __init__(self, *args, **kwargs):
        super(DeepSleepNetTeacher, self).__init__(*args, **kwargs)

        self.model: DeepSleepNetTeacherModel | None = None

    def get_model(self, ):
        self.model = DeepSleepNetTeacherModel(self.name)
        self.model.is_student = False
        self.train()
        return self.model


class DeepSleepNetTeacherModel(DeepSleepNetModelBase):
    def __init__(self, name):
        super(DeepSleepNetTeacherModel, self).__init__(name=name)

     # cnn output 1
        self.conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(6, 1), padding="same",
                            activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3), name="TeacherConv1")
        self.max_pool1 = MaxPooling2D(pool_size=(
            8, 1), strides=(8, 1), padding="same", name="TeacherMaxPool1")

        self.do = Dropout(0.5, name="TeacherDropout1")
        self.conv2 = Conv2D(filters=128, kernel_size=(1, 64),
                            strides=(1, 1), padding="same", activation="relu", name="TeacherConv2")
        self.conv3 = Conv2D(filters=128, kernel_size=(1, 128),
                            strides=(1, 1), padding="same", activation="relu", name="TeacherConv2")
        self.conv4 = Conv2D(filters=128, kernel_size=(1, 128),
                            strides=(1, 1), padding="same", activation="relu", name="TeacherConv2")
        self.max_pool2 = MaxPooling2D(pool_size=(
            4, 1), strides=(4, 1), padding="same", name="TeacherMaxPool2")

        # cnn output 2
        self.conv5 = Conv2D(filters=64, kernel_size=(1, 1),
                            strides=(50, 1), padding="same", activation="relu", name="TeacherConv5")
        self.max_pool3 = MaxPooling2D(pool_size=(
            4, 1), strides=(4, 1), padding="same", name="TeacherMaxPool3")
        self.conv6 = Conv2D(filters=128, kernel_size=(1, 64),
                            strides=(1, 1), padding="same", activation="relu", name="TeacherConv6")
        self.conv7 = Conv2D(filters=128, kernel_size=(1, 128),
                            strides=(1, 1), padding="same", activation="relu", name="TeacherConv6")
        self.conv8 = Conv2D(filters=128, kernel_size=(1, 128),
                            strides=(1, 1), padding="same", activation="relu", name="TeacherConv6")
        self.max_pool4 = MaxPooling2D(pool_size=(
            2, 1), strides=(2, 1), padding="same", name="TeacherMaxPool4")

        # fully connected
        self.fc1 = Dense(1024, activation="relu", name="TeacherFC1")

        # rnn
        self.reshape1 = Reshape(
            target_shape=(-1, 3072), name="teacherFineTuneReshape1")
        self.bidirectional = Bidirectional(
            LSTM(512), merge_mode="concat", name="TeacherBidirectional1")

        # to classes
        self.fc2 = Dense(5, activation="softmax")

    def deep_feature_net_cnn1(self, input):
        output = self.conv1(input)
        output = self.max_pool1(output)
        output = self.do(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.max_pool2(output)
        output = self.flatten(output)
        return output

    def deep_feature_net_cnn2(self, input):
        output = self.conv5(input)
        output = self.max_pool3(output)
        output = self.do(output)
        output = self.conv6(output)
        output = self.conv7(output)
        output = self.conv8(output)
        output = self.max_pool4(output)
        output = self.flatten(output)
        return output

    def deep_sleep_net_fc(self, input):
        return self.fc1(input)

    def deep_sleep_net_rnn(self, input):
        # reshape into (batch_size, seq_length, input_dim)
        output = self.reshape1(input)
        output = self.bidirectional(output)
        return output

    def deep_sleep_net_final_output(self, input):
        return self.fc2(input)
