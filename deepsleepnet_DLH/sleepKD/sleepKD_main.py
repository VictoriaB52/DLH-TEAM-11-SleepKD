from deepsleepnet_DLH.sleepKD.sleepKD_teacher import DeepSleepNetTeacher
from deepsleepnet_DLH.sleepKD.sleepKD_TA import DeepSleepNetTA
from deepsleepnet_DLH.sleepKD.sleepKD_student import DeepSleepNetStudent
from deepsleepnet_DLH.sleepKD.sleepKD_distiller import get_distilled_model
from deepsleepnet_DLH.pretrain_finetune.data_loader import load_all_data as load_all_data_seq
from deepsleepnet_DLH.sleepKD.data_loader import load_all_data
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
import numpy as np


def run_sleepkd_deepsleepnet():
    data_dir = 'data/eeg_fpz_cz/'
    teacher_dir = 'deepsleepnet_DLH/sleepKD/models/deepsleep_teacher.weights.h5'
    student_dir = 'deepsleepnet_DLH/sleepKD/models/deepsleep_student.weights.h5'
    distilled_TA_dir = 'deepsleepnet_DLH/sleepKD/models/deepsleep_distilled_TA.weights.h5'
    distilled_student_dir = 'deepsleepnet_DLH/sleepKD/models/deepsleep_distilled_student.weights.h5'

    batch_size = 20
    training_epochs = 1

    random_state = 12345
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    random.seed(random_state)

    # loads data from all files
    # data, labels = load_all_data(data_dir)

    # split data into train and test sets
    # x_train, x_test, y_train, y_test = train_test_split(
    #     data, labels, test_size=0.1, random_state=random_state)

    # print("\ntraining data:")
    # print("features shape: {}".format(x_train.shape))
    # print("labels shape: {}".format(y_train.shape))
    # print('')
    # print("test data:")
    # print("features shape: {}".format(x_test.shape))
    # print("labels shape: {}".format(y_test.shape))
    # print('')

    # files
    data, labels = load_all_data_seq(data_dir)
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.1, random_state=random_state)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=random_state)

    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # train_dataset = train_dataset.batch(batch_size)

    # val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    # val_dataset = val_dataset.batch(batch_size)

    # teacher model
    print("\nteacher model")

    teacher = DeepSleepNetTeacher(name="TeacherModel", train_dataset=(x_train, y_train), val_dataset=(x_val, y_val),
                                  training_epochs=training_epochs, batch_size=batch_size,
                                  model_dir=teacher_dir)

    teacher_model = teacher.get_model()
    # evaluate(teacher_model, x_test, y_test)

    # teacher_epoch_features = teacher_model.epoch_network
    # teacher_sequence_feautures = teacher_model.sequence_network
    # teacher_soft_labels = teacher_model.logits

    # # student model
    print("\student model")

    student = DeepSleepNetStudent(name="StudentModel", train_dataset=(x_train, y_train), val_dataset=(x_val, y_val),
                                  training_epochs=training_epochs, batch_size=batch_size,
                                  model_dir=student_dir, teacher_model=teacher_model)

    # student_model = student.get_model(teacher_epoch_features=teacher_epoch_features,
    #                                   teacher_sequence_features=teacher_sequence_feautures,
    #                                   teacher_logits=teacher_soft_labels, is_student=True)

    student_model = student.get_model(is_student=True)

    evaluate(student_model, x_test, y_test)

    # # # teacher_assistant distilled from teacher
    # print("\nTA model")

    # teacher_assistant = DeepSleepNetTA(name="UntrainedTA", data=x_train, labels=y_train,
    #                                    training_epochs=training_epochs, batch_size=batch_size,
    #                                    model_dir=None)

    # teacher_assistant_model = teacher_assistant.get_model(should_train=False)

    # distilled_TA_model = get_distilled_model(x_train=x_train,
    #                                          teacher=teacher_model,
    #                                          student_model=teacher_assistant_model,
    #                                          y_train=y_train,
    #                                          training_epochs=1,
    #                                          name="DistilledTAModel",
    #                                          model_dir=distilled_TA_dir)

    # evaluate(distilled_TA_model, x_test, y_test)

    # # student model distilled from TA
    # print("\ndistilled student model")

    # untrained_student = DeepSleepNetStudent(name="UntrainedStudent", data=x_train, labels=y_train,
    #                                         training_epochs=training_epochs, batch_size=batch_size,
    #                                         model_dir=student_dir)
    # untrained_student_model = student.get_model(should_train=False)

    # distilled_student_model = get_distilled_model(x_train=x_train,
    #                                               teacher=distilled_TA_model,
    #                                               student_model=untrained_student_model,
    #                                               y_train=y_train,
    #                                               training_epochs=1,
    #                                               name="DistilledTAModel",
    #                                               model_dir=distilled_student_dir)

    # evaluate(distilled_student_model, x_test, y_test)


def evaluate(model, x_test, y_test):
    preds = model.predict(x_test)
    preds = np.argmax(preds, axis=-1)

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=preds)
    acc = accuracy_score(y_true=y_test, y_pred=preds)
    f1 = f1_score(y_true=y_test,
                  y_pred=preds, average="macro")

    print("\nresults of {}:".format(model.name))
    print(conf_matrix)

    print("accuracy: {}".format(acc))
    print("f1-score: {}".format(f1))
    print('\n')

    return conf_matrix, acc, f1_score
