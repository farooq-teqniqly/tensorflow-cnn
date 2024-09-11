from collections import namedtuple
from tensorflow.keras.datasets import cifar10 as cf10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

_MAX_PIXEL_VALUE=255
METRICS = ["accuracy", Precision(), Recall()]
Dataset = namedtuple("Dataset", ["images", "labels"])

class Cf10:
    def __init__(self):
        pass
        self.training_set:Dataset = None
        self.test_set:Dataset = None

    def load_generate_train_test_sets(self):
        (training_images, training_labels), (test_images, test_labels) = cf10.load_data()
        (training_images, test_images) = Cf10._normalize(training_images, test_images)
        (training_labels, test_labels) = Cf10._one_hot_encode(training_labels, test_labels)
        self.training_set = Dataset(images=training_images, labels=training_labels)
        self.test_set = Dataset(images=test_images, labels=test_labels)

    @classmethod
    def _normalize(cls, training_images, test_images):
        training_images = training_images / _MAX_PIXEL_VALUE
        test_images = test_images / _MAX_PIXEL_VALUE
        return training_images, test_images

    @classmethod
    def _one_hot_encode(cls, training_labels, test_labels):
        training_labels = to_categorical(training_labels)
        test_labels = to_categorical(test_labels)
        return training_labels, test_labels

class CNNBuilder:
    def __init__(self):
        self._model = Sequential()
        self.model_summary = None

    def add_convolution_layer(
            self,
            depth:int,
            kernel_size:int,
            activation="relu",
            **kwargs):

        self._model.add(
            Conv2D(
                depth,
                (kernel_size, kernel_size),
                activation=activation,
                input_shape=kwargs.get("input_shape")))

        return self

    def add_max_pooling_Layer(self, pool_size:int):
        self._model.add(MaxPooling2D((pool_size, pool_size)))
        return self

    def add_flattening_layer(self):
        self._model.add(Flatten())

    def add_fully_connected_layer(
            self,
            units:int,
            activation="relu"):

        self._model.add(Dense(units, activation=activation))
        return self

    def add_output_layer(self, num_classes:int, activation="softmax"):
        self._model.add(Dense(num_classes, activation=activation))
        return self

    def build(self) -> Sequential:
        return self._model

class Compiler:
    def __init__(self, optimizer="adam", loss_function="categorical_crossentropy"):
        self._optimizer = optimizer
        self._loss_function = loss_function

    def compile(self, model:Sequential):
        model.compile(
            optimizer=self._optimizer,
            loss=self._loss_function,
            metrics=METRICS)


class Fitter:
    def __init__(self, model: Sequential):
        self._model = model
        self.training_results = None

    def fit(
            self,
            training_set:Dataset,
            test_set:Dataset,
            epochs:int=30,
            batch_size:int=32):
        self.training_results = self._model.fit(
            training_set.images,
            training_set.labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test_set.images, test_set.labels))

class Predictor:
    def __init__(self, model:Sequential):
        self._model = model

    def predict(self, test_set:Dataset):
        return self._model.predict(test_set.images)

class PerformancePlot:
    def __init__(self, training_results):
        self._training_results = training_results

    def plot_accuracy(self):
        self._plot("accuracy")

    def plot_precision(self):
        self._plot("precision")

    def plot_recall(self):
        self._plot("recall")

    def _plot(
            self,
            metric,
            tolerance=1e-1):
        validation_metric = f"val_{metric}"
        train_perf = self._training_results.history[metric]
        validation_perf = self._training_results.history[validation_metric]

        intersection_index = np.argwhere(np.isclose(
            train_perf, validation_perf, atol=tolerance)).flatten()[0]

        intersection_value = train_perf[intersection_index]

        plt.plot(train_perf, label=metric)
        plt.plot(validation_perf, label=validation_metric)
        plt.axvline(x=intersection_index, color="r", linestyle="--",
                    label="Intersection")

        plt.annotate(
            f"Optimal value: {intersection_value:.2f}",
            xy=(intersection_index, intersection_value),
            xycoords="data",
            fontsize=10,
            color="green")

        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend(loc="lower right")
        plt.show()

class ConfusionMatrix:
    def __init__(self, true_labels, predicted_labels):
        self._true_labels = np.argmax(true_labels, axis=1)
        self._predicted_labels = np.argmax(predicted_labels, axis=1)

    def plot(self):
        cm = confusion_matrix(self._true_labels, self._predicted_labels)
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm)

        cmd.plot(
            include_values=True,
            cmap="inferno",
            ax=None,
            xticks_rotation="horizontal")

        plt.show()
