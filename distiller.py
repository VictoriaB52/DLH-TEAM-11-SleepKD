import tensorflow as tf
import keras
from keras.models import Model
import os


@keras.saving.register_keras_serializable()
class Distiller(keras.Model):
    def __init__(self, student, teacher, name):
        super(Distiller, self).__init__(name=name)
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


def get_distilled_model(x_train, teacher_model, student_model, name=None, y_train=None, training_epochs=1, model_dir=None):

    model = Distiller(teacher=teacher_model, student=student_model, name=name)

    if model_dir and os.path.exists(model_dir):
        model.build(x_train.shape)
        model.load_weights(model_dir)
        return model

    if y_train is None:
        raise Exception(
            "Could not load model and no y_train was provided - can not train new model")

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
        model_dir if model_dir else 'models/deepsleepnet_distilled_weights.h5')
    return model
