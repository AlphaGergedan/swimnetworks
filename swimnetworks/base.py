from __future__ import annotations, division

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator


@dataclass
class Base(BaseEstimator, ABC):
    is_classifier: bool = False
    layer_width: int = None

    activation: Union[Callable[[np.ndarray], np.ndarray], str] = "none"
    weights: np.ndarray = None
    biases: np.ndarray = None
    n_parameters: int = 0

    input_shape: Tuple[int, ...] = None
    output_shape: Tuple[int, ...] = None

    @staticmethod
    def identity_activation(x):
        return x

    @staticmethod
    def relu_activation(x):
        return np.maximum(x, 0)

    @staticmethod
    def tanh_activation(x):
        return np.tanh(x)

    def __post_init__(self):
        self._classes = None

        if not isinstance(self.activation, Callable):
            if self.activation == "none" or self.activation is None:
                self.activation = Base.identity_activation
            elif self.activation == "relu":
                self.activation = Base.relu_activation
            elif self.activation == "tanh":
                self.activation = Base.tanh_activation
            else:
                raise ValueError(f"Unknown activation {self.activation}.")

    @abstractmethod
    def fit(self, x, y=None):
        pass

    def transform(self, x, y=None):
        if self.layer_width is None:
            raise ValueError("The fit method did not set the number of outputs, i.e. layer_width.")

        x = self.prepare_x(x)
        result = self.activation(x @ self.weights + self.biases)
        return result

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)

    def predict(self, x):
        return self.transform(x)

    def prepare_x(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        return x

    def prepare_y(self, y):
        """Prepares labels for the sampling.

        For the classification problem, applies one-hot encoding for the labels.
        For the regression problem, adds a dimension to the labels if neccesary.
        """
        if len(y.shape) < 2:
            y = y.reshape(-1, 1)

        if not self.is_classifier:
            return y, y

        self._classes = np.unique(y)
        n_classes = len(self._classes)
        y_encoded_index = np.argmax(y == self._classes, axis=1)
        y_encoded_onehot = np.eye(n_classes)[y_encoded_index]
        return y_encoded_onehot, y_encoded_index.reshape(-1, 1)


    def prepare_y_inverse(self, y):
        """Inverse to prepare_y(self, y).

        For the classification problem, restores labels from the one-hot predictions.
        For the regression problem, has no effect on the labels.
        """
        if not self.is_classifier:
            return y

        probability_max = np.argmax(y, axis=1)
        predictions = self._classes[probability_max].reshape(-1, 1)
        return predictions

    def clean_inputs(self, x, y=None):
        x = self.prepare_x(x)
        if not y is None:
            y, _ = self.prepare_y(y)
        return x, y
