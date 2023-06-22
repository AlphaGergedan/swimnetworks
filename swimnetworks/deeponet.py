from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

import numpy as np


@dataclass
class DeepONetPOD(BaseEstimator):
    pipeline: Pipeline
    n_modes: int = None
    
    def __post_init__(self):
        self.pod_modes = None 
        self.pod_mean = None

    def fit(self, x, y = None):
        self._set_pod(y)
        pod_y = self._apply_pod(y)
        self.pipeline.fit(x, pod_y)
        return self

    def transform(self, x, y = None):
        prediction = self.pipeline.transform(x)
        restored = self._restore_output(prediction)
        return restored
    
    def _set_pod(self, y):
        mean = y.mean(axis=0)
        shifted = y - mean
        _, _, vh = np.linalg.svd(shifted)
        self.pod_mean = mean
        self.pod_modes = vh.T[:, :self.n_modes]

    def _apply_pod(self, y):
        return (y - self.pod_mean) @ self.pod_modes

    def _restore_output(self, pod_y):
        return pod_y @ self.pod_modes.T + self.pod_mean