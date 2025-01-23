from __future__ import annotations, division

from dataclasses import dataclass

import numpy as np

from .base import Base


@dataclass
class Linear(Base):
    regularization_scale: float = 1e-8
    random_seed: int = -1
    low: float = -np.pi
    high: float = np.pi

    def init_layer(self, dense_layer_width: int):
        """
        Randomly sample layer using normal dist.
        """
        rng = np.random.default_rng(self.random_seed)
        weights, biases = self.sample_parameters_randomly(rng, dense_layer_width)

        self.weights = weights.astype(self.dtype)
        self.biases = biases.astype(self.dtype)

        self.n_parameters = np.prod(weights.shape) + np.prod(biases.shape)
        return self

    def sample_parameters_randomly(self, rng, dense_layer_width):
        print(f"-> sampling weights with size={self.layer_width},{dense_layer_width}")
        weights = rng.normal(loc=0, scale=1, size=(self.layer_width, dense_layer_width)).T
        biases = rng.uniform(low=self.low, high=self.high, size=(self.layer_width)).T
        return weights, biases

    def fit(self, x, y=None):
        if y is None:
            return self

        x, y = self.clean_inputs(x, y)
        # prepare to fit the bias as well
        x = np.column_stack([x, np.ones((x.shape[0], 1))]).astype(self.dtype)
        y = y.astype(self.dtype)

        self.weights = np.linalg.lstsq(x, y, rcond=self.regularization_scale)[0]

        # separate weights and biases
        self.biases = self.weights[-1:, :]
        self.weights = self.weights[:-1, :]
        self.layer_width = self.weights.shape[1]
        self.n_parameters = np.prod(self.weights.shape) + np.prod(self.biases.shape)
        return self

    def transform(self, x, y=None):
        y_predict = super().transform(x, y)
        y_predict = self.prepare_y_inverse(y_predict)
        return y_predict
