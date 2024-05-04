import keras
import time
from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Bidirectional, LSTM

from deepsleepnet_DLH.pretrain_finetune.deepsleepnet_base import DeepSleepNetBase, DeepSleepPreTrainBase


class DeepSleepNetTA(DeepSleepNetBase):
    def __init__(self, *args, **kwargs):
        super(DeepSleepNetTA, self).__init__(*args, **kwargs)

        self.pretrained_model: DeepSleepNetPreTrainTA | None = None
        self.finetuned_model: DeepSleepNetFineTuneTA | None = None

    def get_model(self, train_model=False):

        start_time = time.time()

        # create pretrained model
        self.pretrained_model = DeepSleepNetPreTrainTA(
            name=self.name + "-PreTrain")
        self.finetuned_model = DeepSleepNetFineTuneTA(
            name=self.name + "-FineTune", finetune_batch_size=self.finetune_batch_size, finetune_seq_length=self.finetune_seq_length)

        if train_model:
            self.pretrain()
            self.finetune()

        duration = time.time() - start_time
        print("Took {:.3f}s to train {})".format(duration, self.name))
        return self.pretrained_model, self.finetuned_model


class DeepSleepNetPreTrainTA(DeepSleepPreTrainBase):
    def __init__(self, name):
        super(DeepSleepNetPreTrainTA, self).__init__(name=name)

        # cnn output 1
        self.conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(6, 1), padding="same",
                            activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3), name="TAConv1")
        self.max_pool1 = MaxPooling2D(pool_size=(
            8, 1), strides=(8, 1), padding="same", name="TAMaxPool1")
        self.conv2 = Conv2D(filters=128, kernel_size=(1, 64),
                            strides=(1, 1), padding="same", activation="relu", name="TAConv2")
        self.max_pool2 = MaxPooling2D(pool_size=(
            4, 1), strides=(4, 1), padding="same", name="TAMaxPool2")
        self.reshape1 = Reshape(target_shape=(-1, 2048))

        # cnn output 2
        self.conv3 = Conv2D(filters=64, kernel_size=(1, 1),
                            strides=(50, 1), padding="same", activation="relu", name="TAConv5")
        self.max_pool3 = MaxPooling2D(pool_size=(
            4, 1), strides=(4, 1), padding="same", name="TAMaxPool3")
        self.conv4 = Conv2D(filters=128, kernel_size=(1, 64),
                            strides=(1, 1), padding="same", activation="relu", name="TAConv6")
        self.max_pool4 = MaxPooling2D(pool_size=(
            2, 1), strides=(2, 1), padding="same", name="TAMaxPool4")
        self.reshape2 = Reshape(target_shape=(-1, 1024))

        # output of pretraining - use to calculate loss for model
        self.fc1 = Dense(5, activation="softmax", name='TAPreTrainFC1')

        # will store the network (not including final layer) from forward pass here
        self.network = None

    def deep_feature_net_cnn1(self, input):
        output = self.conv1(input)
        output = self.max_pool1(output)
        output = self.do(output)
        output = self.conv2(output)
        output = self.max_pool2(output)
        output = self.reshape1(output)
        return output

    def deep_feature_net_cnn2(self, input):
        output = self.conv3(input)
        # output = self.relu(self.batch_norm1(self.conv5(input)))
        output = self.max_pool3(output)
        output = self.do(output)
        output = self.conv4(output)
        output = self.max_pool4(output)
        output = self.reshape2(output)
        return output

    def deep_feature_net_final_output(self, input):
        return self.fc1(input)


class DeepSleepNetFineTuneTA(DeepSleepNetPreTrainTA):
    def __init__(self, name: str, finetune_batch_size: int, finetune_seq_length: int):
        super(DeepSleepNetFineTuneTA, self).__init__(name=name)

        self.finetune_batch_size = finetune_batch_size
        self.finetune_seq_length = finetune_seq_length

        # fully connected
        self.fc2 = Dense(512, name="TAFineTuneFC2", activation="relu")

        # rnn
        self.reshape3 = Reshape(target_shape=(-1, 3072),
                                name="TAFineTuneReshape1")
        self.bidirectional = Bidirectional(
            LSTM(256), merge_mode="concat", name="TAFineTuneBidirectional1")

        # dense to classes
        self.fc3 = Dense(5, activation="softmax", name="TAFineTuneFC3")

    def deep_sleep_net_fc(self, input):
        return self.fc2(input)

    def deep_sleep_net_rnn(self, input):
        # reshape into (batch_size, seq_length, input_dim)
        output = self.reshape1(input)
        output = self.bidirectional(output)
        return output

    def deep_sleep_net_final_output(self, input):
        return self.fc3(input)
