import keras
from keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, \
    Dropout, Dense, Bidirectional, LSTM, Add, Concatenate, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import os

from data_loader import iterate_batch_seq_minibatches


class MultiLRAdam(Adam):
    def __init__(self, lr_list):
        self.lr_list = lr_list


class DeepSleepNetTeacher():
    def __init__(self, name, pretrain_data, pretrain_labels,
                 finetune_data, finetune_labels, training_epochs,
                 pretrain_batch_size, finetune_batch_size, finetune_seq_length,
                 pretrained_model_dir, finetuned_model_dir):

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

    def pretrain(self):

        # create model object
        pretrained_model = DeepSleepNetPreTrainTeacher(
            name=self.name + "-PreTrain")

        # if have pre-trained model already, load the weights and return
        if os.path.isfile(self.pretrained_model_dir):
            pretrained_model.build(self.pretrain_data.shape)
            pretrained_model.load_weights(self.pretrained_model_dir)
            return pretrained_model

        # otherwise go through training
        pretrained_model.compile(optimizer=Adam(learning_rate=1e-4),
                                 loss=SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])

        pretrained_model.fit(x=self.pretrain_data, y=self.pretrain_labels, epochs=self.training_epochs, batch_size=self.pretrain_batch_size,
                             validation_split=0.1)

        pretrained_model.save_weights(self.pretrained_model_dir)
        return pretrained_model

    def finetune(self, pretrained_model):
        finetuned_model = DeepSleepNetFineTuneTeacher(
            name=self.name + "-FineTune", finetune_batch_size=self.finetune_batch_size, finetune_seq_length=self.finetune_seq_length)

        finetuned_model.build(
            ((self.finetune_batch_size * self.finetune_seq_length,) + self.pretrain_data.shape[1:]))

        # if have fine-tuned model already, load the weights and return
        if os.path.isfile(self.finetuned_model_dir):
            finetuned_model.load_weights(self.finetuned_model_dir)
            return finetuned_model

        # otherwise go through training

        print(finetuned_model.weights)

        # first load pre-trained weights
        finetuned_model.load_weights(
            self.pretrained_model_dir, by_name=True, skip_mismatch=True)
        # raise NotImplementedError
        print(finetuned_model.weights)

        print("weights loaded")

        # then go through and train entire model, including pre-trained layers
        finetuned_model.compile(optimizer=Adam(learning_rate=1e-4),
                                loss=SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])

        for i in range(0, self.training_epochs):
            for sub_idx, each_data in enumerate(zip(self.finetune_data, self.finetune_labels)):
                each_x, each_y = each_data

                # iterate with minibatches, original batch_size = 10
                for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=each_x,
                                                                      targets=each_y,
                                                                      batch_size=self.finetune_batch_size,
                                                                      seq_length=self.finetune_seq_length):
                    finetuned_model.fit(x_batch, y_batch,
                                        validation_split=0.1)

                print("finished batch {}".format(sub_idx))

        finetuned_model.save_weights(self.finetuned_model_dir)
        return finetuned_model

    def get_model(self,):
        pretrained_model = self.pretrain()
        finetuned_model = self.finetune(pretrained_model)

        return pretrained_model, finetuned_model


class DeepSleepNetPreTrainTeacher(Model):
    def __init__(self, name):
        super(DeepSleepNetPreTrainTeacher, self).__init__(name=name)

        # cnn output 1
        self.conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(
            6, 1), padding="same", kernel_regularizer=keras.regularizers.l2(1e-3), name="teacherPreTrainConv1", activation="relu")
        self.batch_norm1 = BatchNormalization(
            momentum=0.999, epsilon=1e-5, name='teacherPreTrainBatchNorm1')
        self.relu = ReLU()
        self.max_pool1 = MaxPooling2D(pool_size=(
            8, 1), strides=(8, 1), padding="same", name="teacherPreTrainMaxPool1")
        self.do = Dropout(0.5, name="teacherPreTrainDropout1")
        self.conv2 = Conv2D(filters=128, kernel_size=(1, 64),
                            strides=(1, 1), padding="same", name="teacherPreTrainConv2", activation="relu")
        self.batch_norm2 = BatchNormalization(
            momentum=0.999, epsilon=1e-5, name='teacherPreTrainBatchNorm2')
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

        # combine cnn outputs
        self.concat = Concatenate(axis=-1)
        self.flatten = Flatten(name="teacherPreTrainFlatten")
        # output of pretraining - use to calculate loss for model
        self.fc1 = Dense(5, activation="softmax", name='teacherPreTrainFC1')

        # will store the network (not including final layer) from forward pass here
        self.network = None

    def call(self, input):

        # steps of pre-training model in original DSS - Convolution
        cnn1 = self.deep_feature_net_cnn1(input)
        cnn2 = self.deep_feature_net_cnn2(input)
        network = self.concat([cnn1, cnn2])
        network = self.do(network)

        # final layer of pre-training model (in build_model) in original DSS
        network = self.flatten(network)
        print("\nnetwork shape after pre-training: {}".format(network.shape))
        self.network = network

        final_output = self.fc1(network)
        print("final output: {}".format(final_output.shape))

        return final_output

    def deep_feature_net_cnn1(self, input_layer):
        output = self.conv1(input_layer)
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

    def deep_feature_net_cnn2(self, input_var):
        output = self.conv5(input_var)
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


class DeepSleepNetFineTuneTeacher(DeepSleepNetPreTrainTeacher):
    def __init__(self, name, finetune_batch_size, finetune_seq_length):
        super(DeepSleepNetFineTuneTeacher, self).__init__(name=name)

        self.finetune_batch_size = finetune_batch_size
        self.finetune_seq_length = finetune_seq_length

        # fully connected
        self.fc2 = Dense(1024, name="teacherFineTuneFC2", activation="relu")
        self.batch_norm3 = BatchNormalization(
            momentum=0.999, epsilon=1e-5, name='teacherFineTuneBatchNorm3')

        # rnn
        self.reshape1 = Reshape(input_shape=(self.finetune_batch_size * self.finetune_seq_length, 3072),
                                target_shape=(-1, 3072), name="teacherFineTuneReshape1")
        self.bidirectional = Bidirectional(
            LSTM(512), merge_mode="concat", name="teacherFineTuneBidirectional1")

        # combine fc + rnn
        self.add = Add()
        # to classes
        self.fc3 = Dense(5, activation="softmax", name="teacherFineTuneFC3")

    def call(self, input):
        # steps of fine-tuning model in original DSS - RNN

        # pass through pretrained model
        print("calling pre-train with input {}".format(input.shape))
        super(DeepSleepNetFineTuneTeacher, self).call(input)

        print("network shape after pre-train: {}".format(self.network.shape))

        fc = self.deep_sleep_net_fc(self.network)
        rnn = self.deep_sleep_net_rnn(self.network)
        final_output = self.add([fc, rnn])
        final_output = self.do(final_output)

        # soft labels of 5 possible sleep stages
        final_output = self.fc3(final_output)
        print("network shape after fine-tuning: {}".format(final_output.shape))

        return final_output

    def deep_sleep_net_fc(self, input_layer):
        return self.fc2(input_layer)
        # return self.relu(self.batch_norm3(self.fc2(input_layer)))

    def deep_sleep_net_rnn(self, input_layer):
        # reshape into (batch_size, seq_length, input_dim)
        output = self.reshape1(input_layer)
        output = self.bidirectional(output)
        return output


def get_deepsleepnet_teacher_model(name, pretrain_data, pretrain_labels, finetune_data,
                                   finetune_labels, training_epochs, pretrain_batch_size,
                                   finetune_batch_size, finetune_seq_length, pretrained_model_dir=None, finetuned_model_dir=None):

    model = DeepSleepNetTeacher(name=name, pretrain_data=pretrain_data,
                                pretrain_labels=pretrain_labels,
                                finetune_data=finetune_data,
                                finetune_labels=finetune_labels,
                                training_epochs=training_epochs,
                                pretrain_batch_size=pretrain_batch_size,
                                finetune_batch_size=finetune_batch_size,
                                finetune_seq_length=finetune_seq_length,
                                pretrained_model_dir=pretrained_model_dir,
                                finetuned_model_dir=finetuned_model_dir)

    return model
