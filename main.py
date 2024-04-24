import numpy as np
import tensorflow as tf
import keras
import os
import mne
import sklearn
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from data_loader import load_all_data, process_non_seq, iterate_batch_seq_minibatches
from deepsleepnet_DLH.deepsleepnet_teacher import get_deepsleepnet_teacher_model, DeepSleepNetTeacher
from deepsleepnet_DLH.deepsleepnet_student import get_deepsleepnet_student_model
from deepsleepnet_DLH.deepsleepnet_TA import get_deepsleepnet_TA_model
from distiller import get_distilled_model


def main():
    data_dir = 'data/eeg_fpz_cz/'
    deepsleepnet_teacher_pretrained_dir = 'deepsleepnet_DLH/models/TA/deepsleep_teacher_pretrained.weights.h5'
    deepsleepnet_teacher_finetuned_dir = 'deepsleepnet_DLH/models/TA/deepsleep_teacher_finetuned.weights.h5'
    deepsleep_student_weights_dir = 'deepsleepnet_DLH/models/TA/deepsleep_student.weights.h5'
    deepsleep_TA_weights_dir = 'deepsleepnet_DLH/models/TA/deepsleep_TA.weights.h5'
    deepsleep_distilled_student_weights_dir = 'deepsleepnet_DLH/models/TA/deepsleep_distilled.weights.h5'

    pretrain_batch_size = 100
    finetune_batch_size = 10
    finetune_seq_len = 25
    training_epochs = 1

    random_state = 1235

    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    random.seed(random_state)

    # loads lists of data, labels - each spot in list = 1 file's data
    data, labels = load_all_data(data_dir)

    # lists of tensors for training/test files
    x_train_files, x_test_files, y_train_files, y_test_files = train_test_split(
        data, labels, test_size=0.1, random_state=random_state)

    # process training lists into tensors for pre-training (not sequence)
    x_train_non_seq, y_train_non_seq = process_non_seq(
        data=x_train_files, labels=y_train_files)
    x_test_non_seq, y_test_non_seq = process_non_seq(
        data=x_test_files, labels=y_test_files)

    print("non seq training data:")
    print("features shape: {}".format(x_train_non_seq.shape))
    print("labels shape: {}".format(y_train_non_seq.shape))
    print('\n')
    print("test data:")
    print("features shape: {}".format(x_test_non_seq.shape))
    print("labels shape: {}".format(y_test_non_seq.shape))

    # teacher model
    print("\nteacher model")

    # def __init__(self, name, pretrain_data, pretrain_labels, finetune_data,
    #              finetune_labels, training_epochs, pretrain_batch_size, finetune_batch_size, finetune_seq_length):

    deepsleepnet_teacher = DeepSleepNetTeacher(name="TeacherModel", pretrain_data=x_train_non_seq, pretrain_labels=y_train_non_seq,
                                               finetune_data=x_train_files, finetune_labels=y_train_files,
                                               training_epochs=training_epochs, pretrain_batch_size=pretrain_batch_size,
                                               finetune_batch_size=finetune_batch_size, finetune_seq_length=finetune_seq_len,
                                               pretrained_model_dir=deepsleepnet_teacher_pretrained_dir,
                                               finetuned_model_dir=deepsleepnet_teacher_finetuned_dir)

    teacher_pretrained_model, teacher_finetuned_model = deepsleepnet_teacher.get_model()

    teacher_conv_matrix, teacher_acc, teacher_f1_score = evaluate(
        model=teacher_finetuned_model, x_test_files=x_test_files, y_test_files=y_test_files,
        batch_size=finetune_batch_size, seq_length=finetune_seq_len)

    # # student model
    # print("\nstudent model")
    # trained_student_model = get_deepsleepnet_student_model(
    #     x_train=x_train, y_train=y_train, name="StudentModel", return_trained=True, model_dir=deepsleep_student_weights_dir)

    # student_conv_matrix, student_acc, student_f1_score = evaluate(
    #     model=trained_student_model, x_test=x_test, y_test=y_test)

    # # teacher_assistant distilled from teacher
    # print("\nTA model")
    # ta_model = get_deepsleepnet_TA_model(x_train=x_train)

    # distilled_ta_model = get_distilled_model(
    #     x_train=x_train, y_train=y_train, teacher_model=teacher_model, student_model=ta_model, name="DistilledTAModel", model_dir=deepsleep_TA_weights_dir)

    # # student model distilled from TA
    # print("\ndistilled student model")
    # untrained_student_model = get_deepsleepnet_student_model(
    #     x_train, return_trained=False)

    # distilled_student_model = get_distilled_model(x_train=x_train, teacher_model=distilled_ta_model,
    #                                               student_model=untrained_student_model, y_train=y_train, name="DistilledStudentModel", model_dir=deepsleep_distilled_student_weights_dir)

    # distilled_student_preds = distilled_student_model.predict(
    #     x_test, batch_size=100)

    # distilled_student_preds = np.argmax(distilled_student_preds, axis=-1)

    # distilled_student_conv_matrix, distilled_student_acc, distilled_student_f1_score = evaluate(
    #     model=distilled_student_model, x_test=x_test, y_test=y_test)


def evaluate(model, x_test_files, y_test_files, batch_size, seq_length):

    all_preds = []
    all_true = []

    # do mini-batching again, same as when training finetuned model
    for sub_idx, each_data in enumerate(zip(x_test_files, y_test_files)):
        each_x, each_y = each_data
        print(each_y)

        # iterate with minibatches, original batch_size = 10
        for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=each_x,
                                                              targets=each_y,
                                                              batch_size=batch_size,
                                                              seq_length=seq_length):
            preds = model.predict(x_batch)
            preds = np.argmax(preds, axis=-1)
            all_preds.append(preds)
            all_true.append(y_batch)

    all_preds = np.hstack(all_preds)
    all_true = np.hstack(all_true)

    conf_matrix = confusion_matrix(y_true=all_true, y_pred=all_preds)
    acc = accuracy_score(y_true=all_true, y_pred=all_preds)
    f1 = f1_score(y_true=all_true,
                  y_pred=all_preds, average="macro")

    print("\nresults of {}:".format(model.name))
    print(conf_matrix)

    print("accuracy: {}".format(acc))
    print("f1-score: {}".format(f1))
    print('\n')

    return conf_matrix, acc, f1_score


main()
