import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Add, Concatenate, Flatten, Dropout
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from deepsleepnet_DLH.sleepKD.sleepKD import SleepKD
from deepsleepnet_DLH.sleepKD.data_loader import iterate_batch_seq_minibatches
from abc import abstractmethod
from typing import Type
import os
import numpy as np


# superclass for DeepSleepNetTeacher, DeepSleepNetTA, and DeepSleepNetStudent
# handle logic for initializing models, training loops, loading weights if needed
class DeepSleepNetBase():
    def __init__(self, name: str, train_dataset: tuple[list: tf.Tensor, list: tf.Tensor],
                 val_dataset: tuple[list: tf.Tensor, list: tf.Tensor],
                 training_epochs: int, batch_size: int,
                 teacher_model: Type['DeepSleepNetModelBase'] | None = None,
                 model_dir: str = None,):

        self.name = name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.model_dir = model_dir

        self.teacher_model: DeepSleepNetModelBase | None = teacher_model
        self.model: DeepSleepNetModelBase | None = None
        self.training: bool = False
        self.optimizer: Adam = Adam(learning_rate=1e-4)
        self.train_acc_metric = SparseCategoricalAccuracy()
        self.val_acc_metric = SparseCategoricalAccuracy()

        # self.teacher_epoch_network: tf.Tensor | None = None
        # self.teacher_sequence_network: tf.Tensor | None = None

    def train(self):

      # if have pre-trained model already, load the weights and return
        if os.path.isfile(self.model_dir):
            # 0th element of Dataset is TensorSpec for features
            self.model.build(self.train_dataset[0][0].shape)
            self.model.load_weights(self.model_dir)
            return

        # otherwise go through training

        self.model.is_training = True

        # if student, go through distillation training process
        if self.model.is_student:
            for i in range(0, self.training_epochs):
                for sub_idx, each_data in enumerate(zip(*self.train_dataset)):
                    each_x, each_y = each_data

                    # iterate with minibatches, original batch_size = 10
                    for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=each_x,
                                                                          targets=each_y,
                                                                          batch_size=20,
                                                                          seq_length=25):
                        with tf.GradientTape() as tape:
                            logits, sleepKD_loss = self.model.call(
                                x_batch, y_batch)

                        grads = tape.gradient(
                            sleepKD_loss, self.model.trainable_weights)
                        gradient_var_pairs = list(
                            zip(grads, self.model.trainable_weights))
                        self.optimizer.apply_gradients(gradient_var_pairs)

                        self.train_acc_metric.update_state(
                            y_true=y_batch, y_pred=logits)

            # Display metrics at the end of each epoch.
            train_acc = self.train_acc_metric.result()
            print(f"Training acc over epoch: {float(train_acc):.4f}")

            self.train_acc_metric.reset_state()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in self.val_dataset:
                val_logits = self.model(x_batch_val, training=False)
                # Update val metrics
                self.val_acc_metric.update_state(y_batch_val, val_logits)
            val_acc = self.val_acc_metric.result()
            self.val_acc_metric.reset_state()
            print(f"Validation acc: {float(val_acc):.4f}")
            # print(f"Time taken: {time.time() - start_time:.2f}s")

            # need custom training loop so we can pass true_labels for sleepKD layer
            # for epochs in range(self.training_epochs):
            #     for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):

        else:
            self.model.compile(optimizer=Adam(learning_rate=1e-4),
                               loss=SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])
            # self.model.fit(self.train_dataset,
            #                validation_data=self.val_dataset, epochs=self.training_epochs, batch_size=self.batch_size)

            for i in range(0, self.training_epochs):
                for sub_idx, each_data in enumerate(zip(*self.train_dataset)):
                    each_x, each_y = each_data

                    # iterate with minibatches, original batch_size = 10
                    for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=each_x,
                                                                          targets=each_y,
                                                                          batch_size=20,
                                                                          seq_length=25):
                        self.model.fit(x_batch, y_batch,
                                       validation_data=0.1)

                    print("finished batch {}".format(sub_idx))

        self.model.is_training = False

        self.model.save_weights(self.model_dir)


class DeepSleepNetModelBase(Model):
    def __init__(self, name: str, teacher_model: Type['DeepSleepNetModelBase'] | None = None):
        super(DeepSleepNetModelBase, self).__init__(name=name)

        # save as inputs to SleepKD layer
        # self.teacher_epoch_features = teacher_epoch_features
        # self.teacher_sequence_features = teacher_sequence_features
        # self.teacher_logits = teacher_logits
        self.teacher_model = teacher_model
        self.epoch_network = None
        self.sequence_network = None
        self.logits = None

        # know when to call SleepKD layer
        self.is_training = False
        self.is_student = False

        self.concat = Concatenate(axis=-1)
        self.flatten = Flatten()
        self.add = Add()
        self.do = Dropout(0.5)
        self.sleepKD = SleepKD()

    def call(self, input, true_labels=None):

        # steps of pre-training model in original DSS - Convolution
        cnn1 = self.deep_feature_net_cnn1(input)
        print(cnn1.shape)
        cnn2 = self.deep_feature_net_cnn2(input)
        print(cnn2.shape)
        network = self.concat([cnn1, cnn2])
        print(network.shape)
        network = self.do(network)

        # final layer of pre-training model (in build_model) in original DSS
        network = self.flatten(network)
        print(network.shape)
        self.epoch_network = network

        print("\nnetwork shape after pre-training: {}".format(network.shape))

        fc = self.deep_sleep_net_fc(network)
        rnn = self.deep_sleep_net_rnn(network)
        final_output = self.add([fc, rnn])
        final_output = self.do(final_output)
        self.sequence_network = final_output

        print("\noutput shape after finetuning: {}".format(final_output.shape))

        if self.is_training and self.is_student:
            soft_labels = self.deep_sleep_net_final_output(
                self.sequence_network)
            self.teacher_model.is_training = True
            teacher_output = self.teacher_model.call(input=input)
            self.teacher_model.is_training = False

            return soft_labels, self.sleepKD([true_labels, teacher_output, self.teacher_model.epoch_network,
                                             self.epoch_network, self.teacher_model.sequence_network,
                                             self.sequence_network, soft_labels])
        else:
            self.logits = self.deep_sleep_net_final_output(final_output)
            return self.logits

    @abstractmethod
    def deep_feature_net_cnn1(self, input: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def deep_feature_net_cnn2(self, input: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def deep_sleep_net_fc(self, input: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def deep_sleep_net_rnn(self, input: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def deep_sleep_net_final_output(self, input: tf.Tensor) -> tf.Tensor:
        pass
