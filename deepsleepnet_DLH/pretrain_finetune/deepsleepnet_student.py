import keras
import time
from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Bidirectional, LSTM

from deepsleepnet_DLH.pretrain_finetune.deepsleepnet_base import DeepSleepNetBase, DeepSleepPreTrainBase


class DeepSleepNetStudent(DeepSleepNetBase):
    def __init__(self, *args, **kwargs):
        super(DeepSleepNetStudent, self).__init__(*args, **kwargs)

        self.pretrained_model: DeepSleepNetPreTrainStudent | None = None
        self.finetuned_model: DeepSleepNetFineTuneStudent | None = None

    def get_model(self, train_model: bool):

        start_time = time.time()

        self.pretrained_model = DeepSleepNetPreTrainStudent(
            name=self.name + "-PreTrain")
        self.finetuned_model = DeepSleepNetFineTuneStudent(
            name=self.name + "-FineTune", finetune_batch_size=self.finetune_batch_size, finetune_seq_length=self.finetune_seq_length)

        if train_model:
            self.pretrain()
            self.finetune()

        duration = time.time() - start_time
        print("Took {:.3f}s to train {})".format(duration, self.name))

        return self.pretrained_model, self.finetuned_model


class DeepSleepNetPreTrainStudent(DeepSleepPreTrainBase):
    def __init__(self, name):
        super(DeepSleepNetPreTrainStudent, self).__init__(name=name)

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

        # output of pretraining - use to calculate loss for model
        self.fc1 = Dense(5, activation="softmax", name='StudentPreTrainFC1')

        # will store the network (not including final layer) from forward pass here
        self.network = None

    def deep_feature_net_cnn1(self, input):
        output = self.conv1(input)
        output = self.max_pool1(output)
        output = self.do(output)
        output = self.flatten(output)
        return output

    def deep_feature_net_cnn2(self, input):
        output = self.conv2(input)
        output = self.max_pool2(output)
        output = self.do(output)
        output = self.flatten(output)
        return output

    def deep_feature_net_final_output(self, input):
        return self.fc1(input)


class DeepSleepNetFineTuneStudent(DeepSleepNetPreTrainStudent):
    def __init__(self, name: str, finetune_batch_size: int, finetune_seq_length: int):
        super(DeepSleepNetFineTuneStudent, self).__init__(name=name)

        self.finetune_batch_size = finetune_batch_size
        self.finetune_seq_length = finetune_seq_length

        # fully connected
        self.fc2 = Dense(1024, name="StudentFineTuneFC2", activation="relu")

        # rnn
        self.reshape1 = Reshape(input_shape=(self.finetune_batch_size * self.finetune_seq_length, 3072),
                                target_shape=(-1, 3072), name="StudentFineTuneReshape1")
        self.bidirectional = Bidirectional(
            LSTM(128), merge_mode="concat", name="StudentFineTuneBidirectional1")

        # dense to classes
        self.fc3 = Dense(5, activation="softmax", name="StudentFineTuneFC3")

    def deep_sleep_net_fc(self, input):
        return self.fc2(input)

    def deep_sleep_net_rnn(self, input):
        # reshape into (batch_size, seq_length, input_dim)
        print("got input shape {} for reshaping".format(input.shape))
        output = self.reshape1(input)
        output = self.bidirectional(output)
        return output

    def deep_sleep_net_final_output(self, input):
        return self.fc3(input)
