from __future__ import annotations, division

from dataclasses import dataclass
from typing import Callable, Union
import numpy as np

from .base import Base

@dataclass
class Dense(Base):
    parameter_sampler: Union[Callable, str] = "relu"
    sample_uniformly: bool = False
    prune_duplicates: bool = False
    random_seed: int = 1
    dist_min: np.float64 = 1e-10
    repetition_scaler: int = 1

    idx_from: np.ndarray = None
    idx_to: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        self.n_pruned_neurons = 0

        if not isinstance(self.parameter_sampler, Callable):
            if self.parameter_sampler == "relu":
                self.parameter_sampler = self.sample_parameters_relu
            elif self.parameter_sampler == "tanh":
                self.parameter_sampler = self.sample_parameters_tanh
            elif self.parameter_sampler == "random":
                self.parameter_sampler = self.sample_parameters_randomly
            else:
                raise ValueError(f"Unknown parameter sampler {self.parameter_sampler}.")

    def fit(self, x, y=None):
        if self.layer_width is None:
            raise ValueError("layer_width must be set.")

        x, y = self.clean_inputs(x, y)
        rng = np.random.default_rng(self.random_seed)

        weights, biases, idx_from, idx_to = self.parameter_sampler(x, y, rng)

        self.idx_from = idx_from
        self.idx_to = idx_to
        self.weights = weights
        self.biases = biases

        self.n_parameters = np.prod(weights.shape) + np.prod(biases.shape)
        return self

    def sample_parameters_tanh(self, x, y, rng):
        scale = 0.5 * (np.log(1 + 1/2) - np.log(1 - 1/2))

        directions, dists, idx_from, idx_to = self.sample_parameters(x, y, rng)
        weights = (2 * scale * directions / dists).T
        biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1) - scale

        return weights, biases, idx_from, idx_to

    def sample_parameters_relu(self, x, y, rng):
        scale = 1.0

        directions, dists, idx_from, idx_to = self.sample_parameters(x, y, rng)
        weights = (scale / dists.reshape(-1, 1) * directions).T
        biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1)

        return weights, biases, idx_from, idx_to

    def sample_parameters_randomly(self, x, _, rng):
        weights = rng.normal(loc=0, scale=1, size=(self.layer_width, x.shape[1])).T
        biases = rng.uniform(low=-1, high=1, size=(self.layer_width, 1)).T
        idx0 = None
        idx1 = None
        return weights, biases, idx0, idx1

    def sample_parameters(self, x, y, rng):
        """
        Sample directions from points to other points in the given dataset (x, y).
        """

        # n_repetitions repeats the sampling procedure to find better directions.
        # If we require more samples than data points, the repetitions will cause more pairs to be drawn.
        n_repetitions = max(1, int(np.ceil(self.layer_width / x.shape[0]))) * self.repetition_scaler

        # This guarantees that:
        # (a) we draw from all the N(N-1)/2 - N possible pairs (minus the exact idx_from=idx_to case)
        # (b) no indices appear twice at the same position (never idx0[k]==idx1[k] for all k)
        candidates_idx_from = rng.integers(low=0, high=x.shape[0], size=x.shape[0]*n_repetitions)
        delta = rng.integers(low=1, high=x.shape[0]-1, size=candidates_idx_from.shape[0])
        candidates_idx_to = (candidates_idx_from + delta) % x.shape[0]

        directions = x[candidates_idx_to, ...] - x[candidates_idx_from, ...]
        dists = np.linalg.norm(directions, axis=1, keepdims=True)
        dists = np.clip(dists, a_min=self.dist_min, a_max=None)
        directions = directions / dists

        if y is None:
            # data point sampler must be uniformly distributed since no function value information is given,
            # we cannot compute function differences
            assert self.sample_uniformly
            dy = None
        else:
            dy = y[candidates_idx_to, :] - y[candidates_idx_from, :]
            if self.is_classifier:
                dy[np.abs(dy) > 0] = 1

        # We always sample with replacement to avoid forcing to sample low densities
        probabilities = self.weight_probabilities(dists, dy)
        selected_idx = rng.choice(dists.shape[0],
                                  size=self.layer_width,
                                  replace=True,
                                  p=probabilities)

        if self.prune_duplicates:
            selected_idx = np.unique(selected_idx)
            self.n_pruned_neurons = self.layer_width - len(selected_idx)
            self.layer_width = len(selected_idx)

        directions = directions[selected_idx]
        dists = dists[selected_idx]
        idx_from = candidates_idx_from[selected_idx]
        idx_to = candidates_idx_to[selected_idx]

        return directions, dists, idx_from, idx_to


    def weight_probabilities(self, dists, dy=None):
        """Compute probability that a certain weight should be chosen as part of the network.
        This method computes all probabilities at once, without removing the new weights one by one.

        Args:
            dy: function difference
            dists: distance between the base points
            rng: random number generator

        Returns:
            probabilities: probabilities for the weights.
        """
        if self.sample_uniformly:
            probabilities = np.ones(dists.shape[0]) / len(dists)
        else:
            if not dy is None:
                # compute the maximum over all changes in all y directions to sample good gradients for all outputs
                gradients = (np.max(np.abs(dy), axis=1, keepdims=True) / dists).ravel()
                if np.sum(gradients) < self.dist_min:
                    # fallback to uniform sampling
                    probabilities = np.ones(dists.shape[0]) / len(dists)
                else:
                    probabilities = gradients / np.sum(gradients)
            else:
                raise ValueError("Cannot compute gradients without function values.")

        return probabilities
