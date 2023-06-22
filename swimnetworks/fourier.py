from dataclasses import dataclass
from sklearn.base import BaseEstimator

from typing import Sequence, Tuple, Union, Callable
from sklearn.pipeline import Pipeline

from .dense import Dense 
from .linear import Linear

import numpy as np


def crop_rfft_modes(x, ks):
    shift = x.ndim - len(ks)
    slices = [slice(k, -k + 1) for k in ks[:-1]] + [slice(ks[-1], None)]
    cropped = x.copy()
    for i, s in enumerate(slices):
        cropped = np.delete(cropped, s, axis=shift + i)
    return cropped


def pad_rfft_modes(x, target_lengths):
    shift = x.ndim - len(target_lengths)
    indices = [(d + 1) // 2 for d in x.shape[shift:-1]] + [x.shape[-1]]
    last_dim = target_lengths[-1] // 2 + 1
    deltas = [t - d for t, d in zip(target_lengths[:-1], x.shape[shift:-1])] + [
        last_dim - x.shape[-1]
    ]
    padded = x.copy()
    for i, delta in enumerate(deltas):
        ax = shift + i
        padding_shape = np.ones(x.ndim, dtype=np.int64)
        padding_shape[0] = delta
        padding = np.zeros(padding_shape)
        padded = np.insert(padded, indices[i], padding, axis=ax)
    return padded


def rfft(signal, ks_max, norm="backward"):
    """Comptes RFFT along the axes."""
    shift = signal.ndim - len(ks_max)
    transformed = np.fft.rfftn(signal, s=signal.shape[shift:], norm=norm)
    cropped = crop_rfft_modes(transformed, ks_max)
    return cropped


def irfft(modes, target_lengths, norm="backward"):
    """Comptes inverse RFFT along the axes."""
    padded = pad_rfft_modes(modes, target_lengths)
    return np.fft.irfftn(padded, s=target_lengths, norm=norm)


def _split_to_real(x):
    split = np.concatenate([x.real, x.imag], axis=-1)
    return split


def _merge_to_complex(x):
    half = x.shape[-1] // 2
    real = x[..., :half]
    imag = x[..., half:]
    return real + 1j * imag


def _to_int_sequence(parameter):
    if isinstance(parameter, int):
        return [parameter]
    return parameter

@dataclass
class FFT(BaseEstimator):
    ks_max: Sequence = (None,)
    norm: str = "backward"
    avoid_complex: bool = True

    def __post_init__(self):
        self.ks_max = _to_int_sequence(self.ks_max)

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        transformed = rfft(x, self.ks_max, norm=self.norm)
        if self.avoid_complex:
            transformed = _split_to_real(transformed)
        return transformed


@dataclass
class IFFT(BaseEstimator):
    target_lengths: Sequence = (None,)
    norm: str = "backward"
    avoid_complex: bool = True
    real: bool = False

    def __post_init__(self):
        self.target_lengths = _to_int_sequence(self.target_lengths)

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        if self.avoid_complex:
            x = _merge_to_complex(x)
        restored = irfft(x, self.target_lengths, norm=self.norm)
        return restored

@dataclass
class InFourier(BaseEstimator):
    pipeline: Pipeline
    n_modes: int = None
    goal_shape: tuple = (None,)
    avoid_complex: bool = True

    def __post_init__(self):
        self.n_modes = _to_int_sequence(self.n_modes)

    def fit(self, x, y = None):
        fft_transform = FFT(self.n_modes, avoid_complex=self.avoid_complex)
        fft_x = fft_transform.transform(x)
        fft_y = fft_transform.transform(y)
        self.pipeline.fit(fft_x, fft_y)
        self.goal_shape = x.shape[-len(self.n_modes):]

    def transform(self, x, y = None):
        fft_transform = FFT(self.n_modes, avoid_complex=self.avoid_complex)
        fft_x = fft_transform.transform(x)
        prediction = self.pipeline.transform(fft_x)
        ifft_transform = IFFT(self.goal_shape, avoid_complex=self.avoid_complex)
        restored = ifft_transform.transform(prediction)
        return restored 


@dataclass
class Lifting(BaseEstimator):
    n_hidden_channels: int
    random_seed: int
    grid_bounds: Tuple[float, float] = (0, 1)
    grid: np.ndarray = None
    weights: np.ndarray = None
    data_mean: np.ndarray = None 
    data_std: np.ndarray = None

    def _append_grid(self, x):
        expanded_grid = np.repeat(self.grid.reshape(1, -1), len(x), axis=0)
        return np.stack([x, expanded_grid], axis=-1)

    def fit(self, x, y=None):
        if self.grid is None:
            self.grid = np.linspace(*self.grid_bounds, x.shape[-1])

        rng = np.random.default_rng(self.random_seed)
        stacked_x = self._append_grid(x)
        stacked_x = stacked_x.reshape(-1, stacked_x.shape[-1])
        # normalize range
        self.data_mean = np.mean(stacked_x, axis=0, keepdims=True)
        self.data_std = np.std(stacked_x, axis=0, keepdims=True)
        self.weights = rng.uniform(low=-1, high=1,
                                   size=(stacked_x.shape[-1],
                                         self.n_hidden_channels))

    def transform(self, x):
        stacked_x = self._append_grid(x)
        normalized_x = (stacked_x - self.data_mean) / self.data_std
        lifted = normalized_x @ self.weights
        # Put the channel dimension right after the batch one
        return lifted.transpose(0, 2, 1)


@dataclass       
class FourierBlock(Dense, Linear):
    n_hidden_channels: int = None
    n_modes: int = None
    avoid_complex: bool = True
    block_pipelines: Sequence[Pipeline] = None

    def __post_init__(self):
        super().__post_init__()
        rng = np.random.default_rng(self.random_seed)
        self._build_block_pipelines(rng)

    def _build_block_pipelines(self, rng):
        block_pipelines = []
        for _ in range(self.n_hidden_channels):
            random_seed = rng.integers(np.iinfo(np.int64).max)
            steps = [
                ("dense", Dense(layer_width=self.layer_width, activation=self.activation,
                    parameter_sampler=self.parameter_sampler, random_seed=random_seed)),
                ("linear", Linear(regularization_scale=self.regularization_scale))
            ]
            block_pipelines.append(Pipeline(steps))  
        self.block_pipelines = block_pipelines
    
    def fit(self, x, y):
        fft_transform = FFT(self.n_modes, avoid_complex=self.avoid_complex)
        fft_x = fft_transform.transform(x)
        fft_residual = fft_transform.transform(y - x)
        for channel, pipeline in enumerate(self.block_pipelines):
            pipeline.fit(fft_x[:, channel], fft_residual[:, channel])
    
    def transform(self, x, y=None):
        fft_transform = FFT(self.n_modes, avoid_complex=self.avoid_complex)
        goal_shape = x.shape[-self.n_modes:]
        fft_x = fft_transform.transform(x)
        for channel, pipeline in enumerate(self.block_pipelines):
                fft_x[:, channel] = pipeline.transform(fft_x[:, channel])
        ifft_transform = IFFT(goal_shape, avoid_complex=self.avoid_complex)
        restored = ifft_transform.transform(fft_x)
        return restored + x
                

@dataclass
class FNO1D(BaseEstimator):
    n_blocks: int = None
    layer_width: int = None
    n_hidden_channels: int = None
    n_modes: int = None
    random_seed: int = 1
    activation: Union[Callable[[np.ndarray], np.ndarray], str] = "none"
    parameter_sampler: Union[Callable, str] = "relu"
    regularization_scale: float = 1e-8
    avoid_complex: bool = True
    lifting_pipeline: Pipeline = None
    fourier_pipeline: Pipeline = None
    projection_pipeline: Pipeline = None
    
    goal_shape: tuple = (None,)

    def __post_init__(self):
        rng = np.random.default_rng(self.random_seed)

        if self.lifting_pipeline is None:
            random_seed = rng.integers(np.iinfo(np.int64).max)
            self.lifting_pipeline = Lifting(self.n_hidden_channels,
                                            random_seed=random_seed)
        if self.fourier_pipeline is None:
            fourier_steps = []
            for block_id in range(self.n_blocks):
                random_seed = rng.integers(np.iinfo(np.int64).max)
                block = FourierBlock(n_hidden_channels=self.n_hidden_channels,
                                        n_modes=self.n_modes,
                                        layer_width=self.layer_width,
                                        activation=self.activation,
                                        parameter_sampler=self.parameter_sampler,
                                        random_seed=random_seed,
                                        avoid_complex=self.avoid_complex)
                fourier_steps.append((f"fourier{block_id}", block))
            self.fourier_pipeline = Pipeline(fourier_steps)
        
        if self.projection_pipeline is None:
            random_seed = rng.integers(np.iinfo(np.int64).max)
            steps = [
                ("dense", Dense(layer_width=self.layer_width, activation=self.activation,
                    parameter_sampler=self.parameter_sampler, random_seed=random_seed)),
                ("linear", Linear(regularization_scale=self.regularization_scale))
            ]
            self.projection_pipeline = Pipeline(steps)
   
    def fit(self, x, y):
        self.lifting_pipeline.fit(x)
        lifted_x = self.lifting_pipeline.transform(x)
        lifted_y = self.lifting_pipeline.transform(y)
        if self.n_blocks > 0:
            self.fourier_pipeline.fit(lifted_x, lifted_y)
            lifted_x = self.fourier_pipeline.transform(lifted_x)
        lifted_x = lifted_x.reshape(lifted_x.shape[0], -1)
        self.projection_pipeline.fit(lifted_x, y)
    
    def transform(self, x):
        lifted_x = self.lifting_pipeline.transform(x)
        if self.n_blocks > 0:
            lifted_x = self.fourier_pipeline.transform(lifted_x)
        lifted_x = lifted_x.reshape(lifted_x.shape[0], -1)
        return self.projection_pipeline.transform(lifted_x)

        


        

