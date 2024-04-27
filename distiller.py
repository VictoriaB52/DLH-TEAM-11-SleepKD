import tensorflow as tf
import keras
from keras.models import Model
import os
from data_loader import iterate_batch_seq_minibatches


class DistillerBase(Model):
    def __init__(self, student: Model, teacher: Model, name: str):
        super(DistillerBase, self).__init__(name=name)
        self.teacher = teacher
        self.student = student
        self.student_loss_fn = None
        self.distillation_loss_fn = None
        self.alpha: float = None
        self.temperature: float = None

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """Configure the distiller.
        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature


class PreTrainDistiller(DistillerBase):
    def __init__(self, pretrained_teacher: Model, pretrained_student: Model, name: str):
        super(PreTrainDistiller, self).__init__(
            name=name, teacher=pretrained_teacher, student=pretrained_student)

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_pred / self.temperature, axis=1),
            tf.nn.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature**2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        return self.student(x)


def get_distilled_pretrained_model(x_train, y_train, teacher_model, student_model,
                                   name=None, training_epochs=1, model_dir=None):

    model = PreTrainDistiller(pretrained_teacher=teacher_model,
                              pretrained_student=student_model, name=name)

    if model_dir and os.path.exists(model_dir):
        model.build(x_train.shape)
        model.load_weights(model_dir)
        return model

    model.compile(
        optimizer="adam",
        metrics=["accuracy"],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(
            from_logits=False),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    # Distill teacher to student
    model.fit(x_train, y_train, epochs=training_epochs)

    model.save_weights(
        model_dir if model_dir else 'models/TA/deepsleepnet_distilled_pretrained_weights.h5')

    return model


class FineTuneDistiller(DistillerBase):
    def __init__(self, student, teacher, name):
        super(FineTuneDistiller, self).__init__(
            name=name, teacher=teacher, student=student)

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_pred / self.temperature, axis=1),
            tf.nn.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature**2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        return self.student(x)


def get_distilled_finetuned_model(x_train_files: list, y_train_files: list,
                                  teacher_model: Model, student_model: Model,
                                  distilled_pretrain_dir: str, distilled_finetune_dir: str,
                                  batch_size: int, seq_len: int, name: str, training_epochs: int = 1,):

    model = FineTuneDistiller(
        teacher=teacher_model, student=student_model, name=name)

    model.build(
        ((batch_size * seq_len,) + x_train_files[0].shape[1:]))

    if os.path.exists(distilled_finetune_dir):
        model.load_weights(distilled_finetune_dir)
        return model

    model.load_weights(distilled_pretrain_dir,
                       by_name=True, skip_mismatch=True)

    model.compile(
        optimizer="adam",
        metrics=["accuracy"],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(
            from_logits=False),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    for i in range(0, training_epochs):
        for sub_idx, each_data in enumerate(zip(x_train_files, y_train_files)):
            each_x, each_y = each_data

            # iterate with minibatches, original batch_size = 10
            for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=each_x,
                                                                  targets=each_y,
                                                                  batch_size=batch_size,
                                                                  seq_length=seq_len):
                model.fit(x_batch, y_batch,
                          validation_split=0.1)

            print("finished batch {}".format(sub_idx))

    model.save_weights(distilled_finetune_dir)
    return model


def get_distilled_model(x_train: tf.Tensor, x_train_files: list,
                        teacher_pretrain_model: Model, student_pretrain_model: Model,
                        teacher_finetune_model: Model, student_finetune_model: Model,
                        name: str, finetune_batch_size, finetune_seq_len,
                        y_train: list = None, y_train_files: list = None,
                        distilled_pretrain_dir: str = None, distilled_finetune_dir: str = None,
                        training_epochs: int = 1,):

    pretrain_name = name + "-PreTrain"
    finetune_name = name + "FineTune"

    distilled_pretrained_model = get_distilled_pretrained_model(x_train=x_train, y_train=y_train,
                                                                teacher_model=teacher_pretrain_model,
                                                                student_model=student_pretrain_model,
                                                                name=pretrain_name, training_epochs=training_epochs,
                                                                model_dir=distilled_pretrain_dir)

    distilled_finetuned_model = get_distilled_finetuned_model(x_train_files=x_train_files, y_train_files=y_train_files,
                                                              teacher_model=teacher_finetune_model,
                                                              student_model=student_finetune_model,
                                                              distilled_pretrain_dir=distilled_pretrain_dir,
                                                              distilled_finetune_dir=distilled_finetune_dir,
                                                              batch_size=finetune_batch_size, seq_len=finetune_seq_len,
                                                              name=finetune_name, training_epochs=training_epochs)

    return distilled_pretrained_model, distilled_finetuned_model
