"""
The transformations themselves now are objects. This allows for the needed transforms to be precomputed and reused.
There are 5 transforms that it is built for:
    - discrete Phi transform
    - inverse discrete Phi transform
    - short time Phi transform
    - inverse short time Phi transform
    - Vector Voxel transform.
These transformations will be constructed and computed through a GPU.
"""

import numpy as np
import cupy as cp
import dptcomp.constructor as con
import dptcomp.gpu_constructor as gpu_con
from typing import Tuple, List

class DPT_gpu:
    def __init__(self, n: float, time: np.ndarray, nfrequency: np.ndarray, inverse: bool = False):
        """
        Initializes the GPU-based Discrete Phi Transform (DPT) object.

        Input:
            n (float)                   - Chirp order of the transform.
            time (np.ndarray)          - Time-domain array.
            nfrequency (np.ndarray)    - N-frequency domain array.
            inverse (bool)             - Use inverse DPT if True.

        Output:
            DPT_gpu instance.
        """
        self.n = n
        self.inverse = inverse
        self.time = cp.asarray(time)
        self.nfrequency = cp.asarray(nfrequency)
        self.matrices = self._constructTransform()

    def _constructTransform(self) -> List[cp.ndarray]:
        S = cp.asarray(con.construct_S(self.n, cp.asnumpy(self.time), cp.asnumpy(self.nfrequency), self.inverse))
        T = cp.asarray(con.construct_T(len(self.time)))
        return [S, T]

    def transform(self, signal: np.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        signal_gpu = cp.asarray(signal)
        output_signal = self._computeTransform(signal_gpu)
        return output_signal, self.nfrequency

    def _computeTransform(self, signal: cp.ndarray) -> cp.ndarray:
        if signal.shape[0] != len(self.time):
            raise ValueError(f"Invalid input length: {signal.shape[0]} != {len(self.time)}")
        return self.matrices[0] @ (self.matrices[1] @ signal)


class STPT_gpu:
    def __init__(self, n: float, fs: int, nfrequency: np.ndarray, window_size: int):
        """
        Initializes GPU-based Short-Time Phi Transform (STPT).

        Input:
            n (float)                  - Chirp order.
            fs (int)                   - Sampling frequency.
            nfrequency (np.ndarray)    - N-frequency domain.
            window_size (int)          - Window size.

        Output:
            STPT instance.
        """
        self.n = n
        self.fs = fs
        self.nfrequency = cp.asarray(nfrequency)
        self.window_size = window_size
        self.matrices = self._constructTransform()

    def _constructTransform(self) -> List[cp.ndarray]:
        time_window = np.linspace(-self.window_size / (2 * self.fs), self.window_size / (2 * self.fs), self.window_size, endpoint=False)
        S = cp.asarray(con.construct_S(self.n, time_window, cp.asnumpy(self.nfrequency)))
        T = cp.asarray(con.construct_T(self.window_size))
        g = cp.asarray(con.nGauss_wfunc(self.window_size, self.n))
        norm = integrate_vectors(g, g, 1 / self.fs)
        g /= cp.sqrt(norm)
        return [S, T, g]

    def transform(self, signal: np.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        signal_gpu = cp.asarray(signal)
        pixel_array = self._computeTransform(signal_gpu)
        return pixel_array, self.nfrequency

    def _computeTransform(self, signal: cp.ndarray) -> cp.ndarray:
        return self.matrices[0] @ (self.matrices[1] @ gpu_con.constructF(signal, self.window_size, self.matrices[2]))


class ISTPT_gpu:
    def __init__(self, n: float, fs: int, time: np.ndarray, nfrequency: np.ndarray, window_size: int):
        self.n = n
        self.fs = fs
        self.time = cp.asarray(time)
        self.nfrequency = cp.asarray(nfrequency)
        self.window_size = window_size
        self.dt = 1 / self.fs
        self.matrices = self._constructTransform()

    def _constructTransform(self) -> List[cp.ndarray]:
        time_window = np.linspace(-self.window_size / (2 * self.fs), self.window_size / (2 * self.fs), self.window_size, endpoint=False)
        S = cp.asarray(con.construct_S(self.n, cp.asnumpy(self.nfrequency), time_window))
        T = cp.asarray(con.construct_T(len(self.nfrequency)))
        g = cp.asarray(con.nGauss_wfunc(self.window_size, self.n))
        norm_g = integrate_vectors(g, g, self.dt) ** 2
        self.norm_g = norm_g
        return [S, T, g]

    def transform(self, pixel_grid: np.ndarray) -> cp.ndarray:
        pixel_gpu = cp.asarray(pixel_grid)
        signal = self._computeTransform(pixel_gpu)
        return signal / (4 * self.norm_g * self.window_size)

    def _computeTransform(self, pixel_grid: cp.ndarray) -> cp.ndarray:
        """
        Fully GPU-accelerated computation of the ISTPT time-domain signal.
        """
        # Matrix multiplications (still on GPU)
        S = self.matrices[0]  # cp.ndarray
        T = self.matrices[1]  # cp.ndarray
        g = self.matrices[2]  # cp.ndarray
        time_grid = S @ (T @ pixel_grid)

        # Padding
        pad = self.window_size // 2
        padded = cp.hstack((
            cp.zeros((self.window_size, pad), dtype=time_grid.dtype),
            time_grid,
            cp.zeros((self.window_size, pad), dtype=time_grid.dtype)
        ))

        # Vectorized diagonal extraction
        j = cp.arange(self.window_size).reshape(-1, 1)             # shape: (window_size, 1)
        t = cp.arange(len(self.time)).reshape(1, -1)               # shape: (1, num_time_points)
        row_idx = j                                                # (window_size, 1)
        col_idx = j + t                                            # (window_size, num_time_points)
        diag_matrix = padded[row_idx, col_idx]                     # (window_size, num_time_points)

        # Integration over g: matrix dot product with step size
        result = cp.dot(g, diag_matrix) * self.dt                  # shape: (num_time_points,)
        return result


class VVT_gpu:
    def __init__(self, n_range: np.ndarray, fs: int, codomain: np.ndarray, window_size: int):
        self.n_range = n_range
        self.fs = fs
        self.codomain = cp.asarray(codomain)
        self.window_size = window_size
        self.matrices = self._constructTransform()

    def _constructTransform(self):
        time_window = np.linspace(-self.window_size / (2 * self.fs), self.window_size / (2 * self.fs), self.window_size, endpoint=False)
        return [
            (
                n,
                cp.asarray(con.construct_S(n, time_window, cp.asnumpy(self.codomain))),
                cp.asarray(con.construct_T(self.window_size)),
                cp.asarray(con.nGauss_wfunc(self.window_size, n)) / cp.sqrt(integrate_vectors(
                    cp.asarray(con.nGauss_wfunc(self.window_size, n)),
                    cp.asarray(con.nGauss_wfunc(self.window_size, n)),
                    1 / self.fs
                ))
            ) for n in self.n_range
        ]

    def transform(self, signal: np.ndarray) -> Tuple[cp.ndarray, cp.ndarray, np.ndarray]:
        signal_gpu = cp.asarray(signal)
        voxel_grid = self._computeTransform(signal_gpu)
        return voxel_grid, self.codomain, self.n_range

    def _computeTransform(self, signal: cp.ndarray) -> cp.ndarray:
        out = cp.zeros((len(self.codomain), len(signal), len(self.n_range)), dtype=cp.complex64)
        for i, (_, S, T, g) in enumerate(self.matrices):
            out[:, :, i] = S @ (T @ gpu_con.constructF(signal, self.window_size, g))
        return cp.abs(out)


def integrate_vectors(v1: cp.ndarray, v2: cp.ndarray, step_size: float) -> float:
    return cp.dot(v1, v2).item() * step_size


def has_nan(matrix: cp.ndarray) -> bool:
    return cp.isnan(matrix).any().item()


def print_nan_indices(matrix: cp.ndarray):
    indices = cp.argwhere(cp.isnan(matrix))
    if indices.shape[0] > 0:
        print("NaNs at indices:", indices)
    else:
        print("No NaNs found.")