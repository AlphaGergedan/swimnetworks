from __future__ import annotations, division

from dataclasses import dataclass

import numpy as np

from .base import Base


@dataclass
class Linear(Base):
    regularization_scale: float = 1e-8

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
