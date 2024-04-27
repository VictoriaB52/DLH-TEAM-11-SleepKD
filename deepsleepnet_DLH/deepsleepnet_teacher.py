import keras
from keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, \
    Dropout, Dense, Bidirectional, LSTM, Concatenate, Flatten, Reshape
from keras.models import Model

from deepsleepnet_DLH.deepsleepnet_base import DeepSleepNetBase, DeepSleepPreTrainBase


class DeepSleepNetTeacher(DeepSleepNetBase):
    def __init__(self, *args, **kwargs):
        super(DeepSleepNetTeacher, self).__init__(*args, **kwargs)

        self.pretrained_model: DeepSleepNetPreTrainTeacher | None = None
        self.finetuned_model: DeepSleepNetFineTuneTeacher | None = None

    def get_model(self, train_model: bool = True):
        self.pretrained_model = DeepSleepNetPreTrainTeacher(
            name=self.name + "-PreTrain")
        self.finetuned_model = DeepSleepNetFineTuneTeacher(
            name=self.name + "-FineTune", finetune_batch_size=self.finetune_batch_size, finetune_seq_length=self.finetune_seq_length)

        if train_model:
            self.pretrain()
            self.finetune()

        return self.pretrained_model, self.finetuned_model


class DeepSleepNetPreTrainTeacher(DeepSleepPreTrainBase):
    def __init__(self, name):
        super(DeepSleepNetPreTrainTeacher, self).__init__(name=name)

        # cnn output 1
        self.conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(
            6, 1), padding="same", kernel_regularizer=keras.regularizers.l2(1e-3), name="teacherPreTrainConv1", activation="relu")
        self.max_pool1 = MaxPooling2D(pool_size=(
            8, 1), strides=(8, 1), padding="same", name="teacherPreTrainMaxPool1")
        self.do = Dropout(0.5, name="teacherPreTrainDropout1")
        self.conv2 = Conv2D(filters=128, kernel_size=(1, 64),
                            strides=(1, 1), padding="same", name="teacherPreTrainConv2", activation="relu")
        self.conv3 = Conv2D(filters=128, kernel_size=(1, 128),
                            strides=(1, 1), padding="same", name="teacherPreTrainConv3", activation="relu")
        self.conv4 = Conv2D(filters=128, kernel_size=(1, 128),
                            strides=(1, 1), padding="same", name="teacherPreTrainConv4", activation="relu")
        self.max_pool2 = MaxPooling2D(pool_size=(
            4, 1), strides=(4, 1), padding="same", name="teacherPreTrainMaxPool2")

        # cnn output 2
        self.conv5 = Conv2D(filters=64, kernel_size=(1, 1),
                            strides=(50, 1), padding="same", name="teacherPreTrainConv5", activation="relu")
        self.max_pool3 = MaxPooling2D(pool_size=(
            4, 1), strides=(4, 1), padding="same", name="teacherPreTrainMaxPool3")
        self.conv6 = Conv2D(filters=128, kernel_size=(1, 64),
                            strides=(1, 1), padding="same", name="teacherPreTrainConv6", activation="relu")
        self.conv7 = Conv2D(filters=128, kernel_size=(1, 128),
                            strides=(1, 1), padding="same", name="teacherPreTrainConv7", activation="relu")
        self.conv8 = Conv2D(filters=128, kernel_size=(1, 128),
                            strides=(1, 1), padding="same", name="teacherPreTrainConv8", activation="relu")
        self.max_pool4 = MaxPooling2D(pool_size=(
            2, 1), strides=(2, 1), padding="same", name="teacherPreTrainMaxPool4")
        # self.reshape2 = Reshape((-1, 1024), name="teacherReshape2")

        # output of pretraining - use to calculate loss for model
        self.fc1 = Dense(5, activation="softmax", name='teacherPreTrainFC1')

        # will store the network (not including final layer) from forward pass here
        self.network = None

    def deep_feature_net_cnn1(self, input):
        output = self.conv1(input)
        # original DFN used batch normalization between convolution
        # and activation
        # performance took a hit so removed but left call in logic
        # to show where it would've been
        # output = self.relu(self.batch_norm1(self.conv1(input_layer)))
        output = self.max_pool1(output)
        output = self.do(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        # output = self.relu(self.batch_norm2(self.conv2(output)))
        # output = self.relu(self.batch_norm2(self.conv3(output)))
        # output = self.relu(self.batch_norm2(self.conv4(output)))
        output = self.max_pool2(output)
        output = self.flatten(output)
        return output

    def deep_feature_net_cnn2(self, input):
        output = self.conv5(input)
        # output = self.relu(self.batch_norm1(self.conv5(input_var)))
        output = self.max_pool3(output)
        output = self.do(output)
        output = self.conv6(output)
        output = self.conv7(output)
        output = self.conv8(output)
        # output = self.relu(self.batch_norm2(self.conv6(output)))
        # output = self.relu(self.batch_norm2(self.conv7(output)))
        # output = self.relu(self.batch_norm2(self.conv8(output)))
        output = self.max_pool4(output)
        output = self.flatten(output)
        return output

    def deep_feature_net_final_output(self, input_var):
        return self.fc1(input_var)


class DeepSleepNetFineTuneTeacher(DeepSleepNetPreTrainTeacher):
    def __init__(self, name, finetune_batch_size, finetune_seq_length):
        super(DeepSleepNetFineTuneTeacher, self).__init__(name)
        self.finetune_batch_size = finetune_batch_size
        self.finetune_seq_length = finetune_seq_length

        # fully connected
        self.fc2 = Dense(1024, name="teacherFineTuneFC2", activation="relu")
        # self.batch_norm3 = BatchNormalization(
        #     momentum=0.999, epsilon=1e-5, name='teacherFineTuneBatchNorm3')

        # rnn
        self.reshape1 = Reshape(input_shape=(self.finetune_batch_size * self.finetune_seq_length, 3072),
                                target_shape=(-1, 3072), name="teacherFineTuneReshape1")
        self.bidirectional = Bidirectional(
            LSTM(512), merge_mode="concat", name="teacherFineTuneBidirectional1")

        # to classes
        self.fc3 = Dense(5, activation="softmax", name="teacherFineTuneFC3")

    def deep_sleep_net_fc(self, input):
        return self.fc2(input)
        # return self.relu(self.batch_norm3(self.fc2(input)))

    def deep_sleep_net_rnn(self, input):
        # reshape into (batch_size, seq_length, input_dim)
        output = self.reshape1(input)
        output = self.bidirectional(output)
        return output

    def deep_sleep_net_final_output(self, input):
        return self.fc3(input)
