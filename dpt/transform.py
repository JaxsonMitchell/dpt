"""
The transformations themselves now are objects. This allows for the needed transforms to be precomputed and reused.
There are 5 transforms that it is built for.
    - discrete Phi transform
    - inverse discrete Phi transform
    - short time Phi transform
    - inverse short time Phi transform
    - Vector Voxel transform.
"""

import numpy as np
import dptcomp.constructor as con
from typing import Tuple, List
import dpt.readtransform as readtransform


class DPT:  # Both normal and inverse. They are basically the same thing I see no issue
    def __init__(self, n: float, time: np.ndarray, nfrequency: np.array, inverse: bool = False):
        """ 
        Initializes the dpt transform with all necessary
        
        Inputs:
            - n (float) - Chirp Order of the Phi transform such that n > 0.
            - inverse (boolean) - Either True or False. It's automatically 
            - time (np.array) - Time or N-Frequency Values, Pick your poison. 
            - nfrequency (np.array) - The output which is n-frequency values or time. 
                
        Outputs: 
            - Outputs the DPT Object you want to use to transform things :)
        """
        self.n        = n
        self.inverse  = inverse
        self.time   = time
        self.nfrequency = nfrequency
        self.matrices = self._constructTransform()

    def _constructTransform(self) -> List[np.array]:
        S = con.construct_S(self.n, self.time, self.nfrequency, self.inverse)
        T = con.construct_T(len(self.time))
        return [S, T]

    def transform(self, signal: np.array) -> Tuple[np.array, np.array]:
        """ 
        Passes a signal through, it will automatically set the signal 
        """
        output_signal = self._computeTransform(signal)
        return output_signal, self.nfrequency 

    def _computeTransform(self, signal: np.array) -> np.array:
        """
        Computes the transform. In the most general form it takes in an (m, n) signal matrix 
        """
        if np.shape(signal)[0] == len(self.time):
            return np.matmul(self.matrices[0], np.matmul(self.matrices[1], signal))
        else:
            raise ValueError(f"Invalid signal input, domain must be same size as the transform domain. \n"
                             f"{np.shape(signal)[0]} != {len(self.time)}")
        
    def writeToFile(filepath: str):
        """ 
        Writes the DPT Object to an H5 File. 
        """


class STPT:
    def __init__(self, n: float, fs: int, nfrequency: np.array, window_size: np.array):
        """
        Initializes the transform with an additional time delay parameter t_c.
    
        Parameters:
            - n (float)                      - Chirp Order of the transform
            - fs (int)                       - Sampling Frequency of the transform
            - nfrequency: (np.array)         - Frequency domain of the transform
            - window_size: (np.array)        - Size of the nGaussian window
        """
        self.n = n
        self.fs = fs
        self.nfrequency = nfrequency
        self.window_size = window_size
        self.matrices = list(self._constructTransform())

    def _constructTransform(self):
        time_window = np.linspace(
            -self.window_size / (2 * self.fs), 
            self.window_size / (2 * self.fs),
            self.window_size, endpoint=False
        )

        S = con.construct_S(self.n, time_window, self.nfrequency)
        T = con.construct_T(self.window_size)
        g = con.nGauss_wfunc(self.window_size, self.n)

        self.S = S
        self.T = T
        self.g = g / np.sqrt(integrate_vectors(g, g, 1 / self.fs))

        return [S, T, g]

    def transform(self, signal: np.array) -> Tuple[np.array, np.array]: # You should already have the time
        pixel_array = self._computeTransform(signal)
        return pixel_array, self.nfrequency

    def _computeTransform(self, signal) -> np.array:
        return np.matmul(
            self.S, np.matmul(
                self.T, con.constructF(
                    signal, self.window_size, self.g
                )
            )
        )

class ISTPT: # Currently unsure.
    def __init__(self, n: float, time: np.array, codomain: np.array, window_size: np.array):
        self.n = n
        self.time = time
        self.nfrequency = codomain
        self.window_size = window_size
        self.matrices = list(self._constructTransform())

    def _constructTransform(self):
        self.fs = int(1 / (self.time[-1] - self.time[-2]))
        time_window = np.linspace(-self.window_size / (2 * self.fs), self.window_size / (2 * self.fs),
                                  self.window_size, endpoint=False)
        self.dt = 1 / self.fs
        
        S = con.construct_S(self.n, self.nfrequency, time_window)
        T = con.construct_T(len(self.nfrequency))
        g = con.nGauss_wfunc(self.window_size, self.n)
        norm_g = integrate_vectors(g, g, self.dt) ** 2 # It should be normalized, but I don't wanna risk it homie.

        self.S = S
        self.T = T
        self.g = g
        self.norm_g = norm_g

        return [S, T, g]
    
    def _computeTransform(self, pixel_grid: np.array) -> np.array:
        time_grid = np.matmul(
            self.S, np.matmul(self.T, pixel_grid)
        )

        padded_time_grid = np.hstack(
            (
                np.zeros((self.window_size, self.window_size // 2)), 
                time_grid, 
                np.zeros((self.window_size, self.window_size // 2))
            )
        )

        diagonal_vectors = [
            np.array([padded_time_grid[j, j+t] for j in range(self.window_size)]) for t in range(len(self.domain))
        ]

        signal = list(
            map(lambda vector: integrate_vectors(vector, self.g, self.dt), diagonal_vectors)
        ) # I made this a while ago and I'm now thinking to myself... How... did I even do this?

        return signal
    
    def transform(self, pixel_grid: np.array) -> np.array: 
        # I am assuming you have the times that you would want for the signal beforehand. 
        # YOU NEED THAT BTW so not really an assumption Sophomore Jax.
        # Inverse short time transform method. 
        signal = np.array(self._computeTransform(pixel_grid)) / self.norm_g
        return signal


class VVT:
    def __init__(self, n_range: np.arange, fs: np.array, codomain: np.array, window_size: np.array):
        self.n_range = n_range
        self.fs = fs
        self.codomain = codomain
        self.window_size = window_size
        self.matrices = list(self._constructTransform())

    def _constructTransform(self):
        time_window = np.linspace(-self.window_size / (2 * self.fs), self.window_size / (2 * self.fs),
                                  self.window_size, endpoint=False)
        return [(
            n,
            con.construct_S(n, time_window, self.codomain),
            con.construct_T(self.window_size),
            con.nGauss_wfunc(self.window_size, n) / np.sqrt(integrate_vectors(
                con.nGauss_wfunc(self.window_size, n), con.nGauss_wfunc(self.window_size, n), 1 / self.fs
            ))
        ) for n in self.n_range]

    def transform(self, signal: np.array) -> Tuple[np.array, np.array, np.array]:
        voxel_array = self._computeTransform(signal)
        return voxel_array, self.codomain, self.n_range

    def _computeTransform(self, signal: np.array) -> np.array:
        voxel_grid = np.zeros((len(self.codomain), len(signal), len(self.n_range)), dtype=complex)
        for i in range(len(self.n_range)):
            S = self.matrices[i][1]
            T = self.matrices[i][2]
            g = self.matrices[i][3]
            voxel_grid[:, :, i] = np.matmul(
                S, np.matmul(
                    T, con.constructF(
                        signal, self.window_size, g
                    )
                )
            )
        return np.abs(voxel_grid)

# These were used to correct some potential not a number errors that I was getting at some point
def has_nan(matrix):
    return np.isnan(matrix).any()


def print_nan_indices(matrix):
    indices = np.argwhere(np.isnan(matrix))
    if len(indices) > 0:
        print("NaN values found at the following indices:")
        for index in indices:
            print(index)
    else:
        print("No NaN values found in the matrix.")

# Used within my ISTPT code.
def integrate_vectors(v1: np.array, v2: np.array, step_size):
    """ This utilizes the dot product for efficient 
    integration given a constant step size. This
    is utilized by the ISTPT transform method."""
    return np.dot(v1, v2) * step_size

if __name__ == "__main__":
    n = 2
    times = np.arange(0, 1, 1/1024)
    freqs = np.arange(0, 32, 1/8)

    dpt   = DPT(n, times, freqs)