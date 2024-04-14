import numpy as np
import tensorflow as tf
import keras
import os
import mne
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from data_loader import load_all_data
from deepsleep_teacher import get_deep_sleep_teacher_model
from deepsleep_student import get_deep_sleep_student_model


@keras.saving.register_keras_serializable()
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.student_loss_fn = None
        self.distillation_loss_fn = None
        self.alpha = None
        self.temperature = None

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "student": keras.saving.serialize_keras_object(self.student),
            "teacher": keras.saving.serialize_keras_object(self.teacher),
        })
        return config

    def get_compile_config(self):
        # serialize params used in compilation
        return {
            "student_loss_fn": keras.saving.serialize_keras_object(self.student_loss_fn),
            "distillation_loss_fn": keras.saving.serialize_keras_object(self.distillation_loss_fn),
            "alpha": keras.saving.serialize_keras_object(self.alpha),
            "temperature": keras.saving.serialize_keras_object(self.temperature),
            "optimizer": keras.saving.serialize_keras_object(self.optimizer),
            "metrics": keras.saving.serialize_keras_object(self.metrics),
        }

    @classmethod
    def from_config(cls, config):
        student_config = config.pop("student")
        teacher_config = config.pop("teacher")

        student = keras.saving.deserialize_keras_object(student_config)
        teacher = keras.saving.deserialize_keras_object(teacher_config)
        return cls(student, teacher, **config)

    def compile_form_config(self, config):
        student_loss_fn_config = config.pop("student_loss_fn")
        distillation_loss_fn_config = config.pop("distillation_loss_fn")
        alpha_config = config.pop("alpha")
        temperature_config = config.pop("temperature")
        optimizer_config = config.pop("optimizer")
        metrics_config = config.pop("metrics")

        student_loss_fn = keras.saving.deserialize_keras_object(
            student_loss_fn_config)
        distillation_loss_fn = keras.saving.deserialize_keras_object(
            distillation_loss_fn_config)
        alpha = keras.saving.deserialize_keras_object(alpha_config)
        temperature = keras.saving.deserialize_keras_object(temperature_config)
        optimizer = keras.saving.deserialize_keras_object(optimizer_config)
        metrics = keras.saving.deserialize_keras_object(metrics_config)
        self.compile(optimizer, metrics, student_loss_fn,
                     distillation_loss_fn, alpha, temperature)


def main():
    DATA_DIR = 'data/eeg_fpz_cz/'
    TEACHER_MODEL_DIR = 'deep_sleep_net_teacher.keras'
    STUDENT_MODEL_DIR = 'deep_sleep_net_student.keras'
    DISTILLED_STUDENT_DIR = 'deep_sleep_net_distilled.keras'

    print(tf.__version__)
    print(keras.__version__)
    print(np.__version__)
    print(mne.__version__)
    print(sklearn.__version__)

    np.random.seed(12345)
    tf.random.set_seed(12345)

    data, labels = load_all_data(DATA_DIR)

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.1, random_state=12345)

    teacher_model = get_deep_sleep_teacher_model(
        TEACHER_MODEL_DIR, x_train, y_train)

    teacher_preds = teacher_model.predict(x_test, batch_size=100)
    teacher_preds = np.argmax(preds, axis=1)

    conf_matrix = confusion_matrix(y_test, teacher_preds)
    print(conf_matrix)
    print(accuracy_score(y_test, teacher_preds))
    print(f1_score(y_test, teacher_preds, average="macro"))

    student_model = get_deep_sleep_student_model(STUDENT_MODEL_DIR, x_train)
    distiller = None
    if os.path.isfile(DISTILLED_STUDENT_DIR):
        distiller = keras.models.load_model(DISTILLED_STUDENT_DIR)
    else:
        distiller = Distiller(student=student_model, teacher=teacher_model)

        distiller.compile(
            optimizer="adam",
            metrics=["accuracy"],
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=10,
        )

        # Distill teacher to student
        distiller.fit(x_train, y_train, epochs=1)
        distiller.save(DISTILLED_STUDENT_DIR)

    # Evaluate student on test dataset
    student_preds = distiller.predict(x_test, batch_size=100)

    student_preds = np.argmax(student_preds, axis=1)

    conf_matrix = confusion_matrix(y_test, student_preds)
    print(conf_matrix)
    print(accuracy_score(y_test, student_preds))
    print(f1_score(y_test, student_preds, average="macro"))


main()


#  FINE TUNED MODEL ONLY CODE - KEEP JUST IN CASE

# def get_finetuned_model(FINE_TUNED_MODEL_DIR, PRE_TRAINED_MODEL_DIR, DATA_DIR, n_folds=20, fold_idx=0):

#     if os.path.isfile(FINE_TUNED_MODEL_DIR):
#         return keras.models.load_model(FINE_TUNED_MODEL_DIR)

#     pretrained_model = get_pretrained_model(
#         PRE_TRAINED_MODEL_DIR, DATA_DIR, n_folds, fold_idx)

#     data_loader = SeqDataLoader(
#         data_dir=DATA_DIR,
#         n_folds=n_folds,
#         fold_idx=fold_idx)

#     x_train, y_train, x_valid, y_valid = data_loader.load_train_data()

#     model = deep_sleep_net_model(
#         pretrained_model.input, pretrained_model.output)

#     model.compile(optimizer='adam',
#                   loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

#     model.fit(x_train, y_train, epochs=1, batch_size=100,
#               validation_data=(x_valid, y_valid))

#     model.save(FINE_TUNED_MODEL_DIR)
#     return model
