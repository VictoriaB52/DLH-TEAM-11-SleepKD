from deepsleepnet_DLH.pretrain_finetune.deepsleepnet_teacher import DeepSleepNetTeacher
from deepsleepnet_DLH.pretrain_finetune.deepsleepnet_TA import DeepSleepNetTA
from deepsleepnet_DLH.pretrain_finetune.deepsleepnet_student import DeepSleepNetStudent
from deepsleepnet_DLH.pretrain_finetune.deepsleepnet_distiller import get_distilled_model
from deepsleepnet_DLH.pretrain_finetune.data_loader import load_all_data, process_non_seq, iterate_batch_seq_minibatches
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
import numpy as np


def run_pretrain_finetune_deepsleepnet():
    data_dir = 'data/eeg_fpz_cz/'
    teacher_pretrained_dir = 'deepsleepnet_DLH/pretrain_finetune/models/deepsleep_teacher_pretrained.weights.h5'
    teacher_finetuned_dir = 'deepsleepnet_DLH/pretrain_finetune/models/deepsleep_teacher_finetuned.weights.h5'
    student_pretrained_dir = 'deepsleepnet_DLH/pretrain_finetune/models/deepsleep_student_pretrained.weights.h5'
    student_finetuned_dir = 'deepsleepnet_DLH/pretrain_finetune/models/deepsleep_student_finetuned.weights.h5'
    distilled_TA_pretrained = 'deepsleepnet_DLH/pretrain_finetune/models/deepsleep_distilled_TA_pretrained.weights.h5'
    distilled_TA_finetuned = 'deepsleepnet_DLH/pretrain_finetune/models/deepsleep_distilled_TA_finetuned.weights.h5'
    distilled_student_pretrained = 'deepsleepnet_DLH/pretrain_finetune/models/deepsleep_distilled_student_pretrained.weights.h5'
    distilled_student_finetuned = 'deepsleepnet_DLH/pretrain_finetune/models/deepsleep_distilled_student_finetuned.weights.h5'

    pretrain_batch_size = 20
    finetune_batch_size = 20
    finetune_seq_len = 25
    training_epochs = 1

    random_state = 12345

    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    random.seed(random_state)

    # loads lists of data, labels - each spot in list = 1 file's data
    data, labels = load_all_data(data_dir)

    # lists of tensors for training/test files
    x_train_files, x_test_files, y_train_files, y_test_files = train_test_split(
        data, labels, test_size=0.1, random_state=random_state)

    # process training lists into tensors for pre-training (cnn - not sequences)
    x_train_non_seq, y_train_non_seq = process_non_seq(
        data=x_train_files, labels=y_train_files)
    # process testing lists into tensors for evaluation
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

    teacher = DeepSleepNetTeacher(name="TeacherModel", pretrain_data=x_train_non_seq, pretrain_labels=y_train_non_seq,
                                  finetune_data=x_train_files, finetune_labels=y_train_files,
                                  training_epochs=training_epochs, pretrain_batch_size=pretrain_batch_size,
                                  finetune_batch_size=finetune_batch_size, finetune_seq_length=finetune_seq_len,
                                  pretrained_model_dir=teacher_pretrained_dir,
                                  finetuned_model_dir=teacher_finetuned_dir)

    pretrained_teacher_model, finetuned_teacher_model = teacher.get_model()

    # teacher_conv_matrix, teacher_acc, teacher_f1_score = evaluate(
    #     model=finetuned_teacher_model, x_test_files=x_test_files, y_test_files=y_test_files,
    #     batch_size=finetune_batch_size, seq_length=finetune_seq_len)

    # # student model
    # print("\nstudent model")
    # student = DeepSleepNetStudent(
    #     name="TrainedStudentModel", pretrain_data=x_train_non_seq, pretrain_labels=y_train_non_seq,
    #     finetune_data=x_train_files, finetune_labels=y_train_files,
    #     training_epochs=training_epochs, pretrain_batch_size=pretrain_batch_size,
    #     finetune_batch_size=finetune_batch_size, finetune_seq_length=finetune_seq_len,
    #     pretrained_model_dir=student_pretrained_dir,
    #     finetuned_model_dir=student_finetuned_dir)

    # _, trained_student_finetuned_model = student.get_model(
    #     train_model=True)

    # student_conv_matrix, student_acc, student_f1_score = evaluate(
    #     model=trained_student_finetuned_model, x_test_files=x_test_files, y_test_files=y_test_files,
    #     batch_size=finetune_batch_size, seq_length=finetune_seq_len)

    # # teacher_assistant distilled from teacher
    print("\nTA model")
    teacher_assistant = DeepSleepNetTA(
        name="TAModel", pretrain_data=x_train_non_seq, pretrain_labels=y_train_non_seq,
        finetune_data=x_train_files, finetune_labels=y_train_files,
        training_epochs=training_epochs, pretrain_batch_size=pretrain_batch_size,
        finetune_batch_size=finetune_batch_size, finetune_seq_length=finetune_seq_len)

    untrained_pretrain_TA_model, untrained_finetune_TA_model = teacher_assistant.get_model(
        train_model=False)

    distilled_TA_pretrain_model, distilled_TA_finetune_model = get_distilled_model(
        x_train=x_train_non_seq, x_train_files=x_train_files,
        y_train=y_train_non_seq, y_train_files=y_train_files,
        teacher_pretrain_model=pretrained_teacher_model,
        teacher_finetune_model=finetuned_teacher_model,
        student_pretrain_model=untrained_pretrain_TA_model,
        student_finetune_model=untrained_finetune_TA_model,
        finetune_batch_size=finetune_batch_size,
        finetune_seq_len=finetune_seq_len,
        name="DistilledTAModel",
        distilled_pretrain_dir=distilled_TA_pretrained,
        distilled_finetune_dir=distilled_TA_finetuned,
    )

    teacher_assistant_conv_matrix, teacher_assistant_acc, teacher_assistant_f1_score = evaluate(
        model=distilled_TA_finetune_model, x_test_files=x_test_files, y_test_files=y_test_files,
        batch_size=finetune_batch_size, seq_length=finetune_seq_len)

    # student model distilled from TA
    print("\ndistilled student model")

    untrained_student = DeepSleepNetStudent(name="UntrainedStudentModel",
                                            pretrain_data=x_train_non_seq, pretrain_labels=y_train_non_seq,
                                            finetune_data=x_train_files, finetune_labels=y_train_files,
                                            training_epochs=training_epochs, pretrain_batch_size=pretrain_batch_size,
                                            finetune_batch_size=finetune_batch_size, finetune_seq_length=finetune_seq_len)

    untrained_pretrain_student_model, untrained_finetune_student_model = untrained_student.get_model(
        train_model=False)

    _, distilled_student_finetune_model = get_distilled_model(
        x_train=x_train_non_seq, x_train_files=x_train_files,
        y_train=y_train_non_seq, y_train_files=y_train_files,
        teacher_pretrain_model=distilled_TA_pretrain_model,
        teacher_finetune_model=finetuned_teacher_model,
        student_pretrain_model=finetuned_teacher_model,
        student_finetune_model=untrained_finetune_TA_model,
        finetune_batch_size=finetune_batch_size,
        finetune_seq_len=finetune_seq_len,
        name="DistilledStudentModel",
        distilled_pretrain_dir=distilled_student_pretrained,
        distilled_finetune_dir=distilled_student_finetuned,
    )

    distilled_student_conv_matrix, distilled_student_acc, distilled_student_f1_score = evaluate(
        model=distilled_student_finetune_model, x_test_files=x_test_files, y_test_files=y_test_files,
        batch_size=finetune_batch_size, seq_length=finetune_seq_len)


def evaluate(model, x_test_files, y_test_files, batch_size, seq_length):

    all_preds = []
    all_true = []

    # do mini-batching again, same as when training finetuned model
    for sub_idx, each_data in enumerate(zip(x_test_files, y_test_files)):
        each_x, each_y = each_data

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
