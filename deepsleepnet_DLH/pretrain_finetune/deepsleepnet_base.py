import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Add, Concatenate, Flatten, Dropout
from keras.losses import SparseCategoricalCrossentropy
from abc import abstractmethod
import os


from deepsleepnet_DLH.pretrain_finetune.data_loader import iterate_batch_seq_minibatches


# superclass for DeepSleepNetTeacher, DeepSleepNetTA, and DeepSleepNetStudent
# handle logic for initializing models, training loops, loading weights if needed
class DeepSleepNetBase():
    def __init__(self, name: str, pretrain_data: tf.Tensor, pretrain_labels: tf.Tensor,
                 finetune_data: list, finetune_labels: list, training_epochs: int,
                 pretrain_batch_size: int, finetune_batch_size: int, finetune_seq_length: int,
                 pretrained_model_dir: str = None, finetuned_model_dir: str = None):

        self.name = name
        self.pretrain_data = pretrain_data
        self.pretrain_labels = pretrain_labels
        self.finetune_data = finetune_data
        self.finetune_labels = finetune_labels
        self.training_epochs = training_epochs
        self.pretrain_batch_size = pretrain_batch_size
        self.finetune_batch_size = finetune_batch_size
        self.finetune_seq_length = finetune_seq_length
        self.pretrained_model_dir = pretrained_model_dir
        self.finetuned_model_dir = finetuned_model_dir

        self.pretrained_model: Model | None = None
        self.finetuned_model: Model | None = None

    def pretrain(self):

        # if have pre-trained model already, load the weights and return
        if os.path.isfile(self.pretrained_model_dir):
            self.pretrained_model.build(self.pretrain_data.shape)
            self.pretrained_model.load_weights(self.pretrained_model_dir)
            return

        # otherwise go through training
        self.pretrained_model.compile(optimizer=Adam(learning_rate=1e-4),
                                      loss=SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])

        self.pretrained_model.fit(x=self.pretrain_data, y=self.pretrain_labels, epochs=self.training_epochs, batch_size=self.pretrain_batch_size,
                                  validation_split=0.1)

        self.pretrained_model.save_weights(self.pretrained_model_dir)

    def finetune(self):

        print(self.finetuned_model.summary)

        self.finetuned_model.build(
            ((self.finetune_batch_size * self.finetune_seq_length,) + self.pretrain_data.shape[1:]))

        # if have fine-tuned model already, load the weights and return
        if os.path.isfile(self.finetuned_model_dir):
            self.finetuned_model.load_weights(
                self.finetuned_model_dir, by_name=True)
            return

        # otherwise go through training

        # first load pre-trained weights
        self.finetuned_model.load_weights(
            self.pretrained_model_dir, by_name=True, skip_mismatch=True)

        # then go through and train entire model, including pre-trained layers
        # actual DeepSleepNet uses lower learning rate for pre-trained layers
        self.finetuned_model.compile(optimizer=Adam(learning_rate=1e-4),
                                     loss=SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])

        for i in range(0, self.training_epochs):
            for sub_idx, each_data in enumerate(zip(self.finetune_data, self.finetune_labels)):
                each_x, each_y = each_data

                # iterate with minibatches, original batch_size = 10
                for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=each_x,
                                                                      targets=each_y,
                                                                      batch_size=self.finetune_batch_size,
                                                                      seq_length=self.finetune_seq_length):
                    self.finetuned_model.fit(x_batch, y_batch,
                                             validation_split=0.1)

                print("finished batch {}".format(sub_idx))

        self.finetuned_model.save_weights(self.finetuned_model_dir)


# superclass for pre-train model of teacher, TA, studen
class DeepSleepPreTrainBase(Model):
    def __init__(self, name):
        super(DeepSleepPreTrainBase, self).__init__(name=name)
        self.concat = Concatenate(axis=-1)
        self.flatten = Flatten()
        self.add = Add()
        self.do = Dropout(0.5)

    def call(self, input):

        # steps of pre-training model in original DSS - Convolution
        print("pre-train input: {}".format(input.shape))
        cnn1 = self.deep_feature_net_cnn1(input)
        print(cnn1.shape)
        cnn2 = self.deep_feature_net_cnn2(input)
        print(cnn2.shape)
        network = self.concat([cnn1, cnn2])
        print(network.shape)
        network = self.do(network)

        # final layer of pre-training model (in build_model) in original DSS
        network = self.flatten(network)
        print("\nnetwork shape after pre-training: {}".format(network.shape))
        self.network = network

        final_output = self.deep_feature_net_final_output(network)
        print("final output shape: {}\n".format(final_output.shape))

        return final_output

    @abstractmethod
    def deep_feature_net_cnn1(self, input: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def deep_feature_net_cnn2(self, input: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def deep_feature_net_final_output(self, input: tf.Tensor) -> tf.Tensor:
        pass


class DeepSleepNetFineTuneBase(DeepSleepPreTrainBase):
    def __init__(self, name):
        super(DeepSleepNetFineTuneBase, self).__init__(name=name)

    def call(self, input):
        # steps of fine-tuning model in original DSS - RNN

        # pass through pretrained model
        print("finetune call input: {}".format(input.shape))
        super(DeepSleepNetFineTuneBase, self).call(input)

        print("network shape after pre-train: {}".format(self.network.shape))

        fc = self.deep_sleep_net_fc(self.network)
        print("rnn input shape {}".format(self.network.shape))
        rnn = self.deep_sleep_net_rnn(self.network)
        print("rnn output shape {}".format(rnn.shape))
        final_output = self.add([fc, rnn])
        final_output = self.do(final_output)

        # soft labels of 5 possible sleep stages
        final_output = self.deep_sleep_net_final_output(final_output)

        print("network shape after fine-tuning: {}".format(final_output.shape))

        return final_output

    @abstractmethod
    def deep_sleep_net_fc(self, input: tf.Tensor):
        pass

    @abstractmethod
    def deep_sleep_net_rnn(self, input: tf.Tensor):
        pass

    @abstractmethod
    def deep_sleep_net_final_output(self, input: tf.Tensor):
        pass
