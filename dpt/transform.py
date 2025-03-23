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
from dptcomp.sgn import Signal
from dptcomp.pixel import PixelGrid, VoxelGrid
from tqdm.auto import tqdm
from abc import abstractmethod


class TransBC:  # Transform Base Class
    def __init__(self, domain):
        self.domain = domain

    @abstractmethod
    def _constructTransform(self):
        pass

    @abstractmethod
    def transform(self, argument: Signal or PixelGrid):
        pass

    @abstractmethod
    def _computeTransform(self, argument: Signal or PixelGrid):
        pass


class DPT(TransBC):  # Both normal and inverse.
    def __init__(self, n: float, domain: np.array, codomain: np.array, inverse: bool = False):
        """ Initializes the dpt transform with all necessary """
        super().__init__(domain)
        self.n = n
        self.inverse = inverse
        self.codomain = codomain
        self.matrices = self._constructTransform()

    def _constructTransform(self) -> list[np.array]:
        S = con.construct_S(self.n, self.domain, self.codomain, self.inverse)
        T = con.construct_T(len(self.domain))
        return [S, T]

    def transform(self, signal: Signal) -> Signal:
        """ Passes a signal through, it will automatically set the signal """
        w = self._computeTransform(signal)
        output_signal = Signal(self.codomain, w)
        self._labelSignal(output_signal)
        return output_signal

    def _computeTransform(self, signal: Signal) -> np.array:
        """
        Computes the transform.
        """
        if len(signal.domain) == len(self.domain):
            return np.matmul(self.matrices[0], np.matmul(self.matrices[1], signal.amplitude))
        else:
            raise ValueError(f"Invalid signal input, domain must be same size as the transform domain. \n"
                             f"{len(signal.domain)} != {len(self.domain)}")

    def _labelSignal(self, signal: Signal) -> None:
        if not self.inverse:
            signal.labelSignal("transformed signal", "frequency", "amplitude")
        else:
            signal.labelSignal("signal")


class STPT(TransBC):
    def __init__(self, n: float, domain: np.array, codomain: np.array, window_size: np.array, time_delay: float = 0.0):
        """
            Initializes the transform with an additional time delay parameter t_c.
    
            Parameters:
            - n (float)                      - Chirp Order of the transform
            - domain (np.array)              - Time domain of the transform
            - codomain: (np.array)           - Frequency domain of the transform
            - window_size: (np.array)        - Size of the nGaussian window
            - time_delay: (float, optional)  - Time shift parameter
        """
        self.n = n
        self.domain = domain
        self.codomain = codomain
        self.window_size = window_size
        self.matrices = list(self._constructTransform())
        self.time_day = time_delay

    def _constructTransform(self):
        # If you aren't using an integer sampling frequency, please... what? Fix this if that's the case. I don't quite care enough.
        fs = int(1 / (self.domain[-1] - self.domain[-2]))  
        time_window = np.linspace(-self.window_size / (2 * fs), self.window_size / (2 * fs),
                                  self.window_size, endpoint=False)  # Centered about zero, but luckily this doesn't change the time shift issue

        S = con.construct_S(self.n, time_window - self.time_delay, self.codomain)
        T = con.construct_T(self.window_size)
        g = con.nGauss_wfunc(self.window_size, self.n)

        self.S = S
        self.T = T
        self.g = g

        return [S, T, g]

    def transform(self, signal: Signal) -> PixelGrid:
        pixel_array = self._computeTransform(signal)
        output_pixelgrid = PixelGrid(
            time=self.domain,
            frequency=self.codomain,
            gridValue=pixel_array,
            n=self.n
        )
        return output_pixelgrid

    def _computeTransform(self, signal: Signal) -> np.array:
        if len(signal.domain) == len(self.domain):
            return np.matmul(
                self.S, np.matmul(
                    self.T, con.constructF(
                        signal.amplitude, self.window_size, self.g
                    )
                )
            )
        else:
            raise ValueError(f"Invalid signal input, domain must be same size as the transform domain. \n"
                             f"{len(signal.domain)} != {len(self.domain)}")


class ISTPT(TransBC):
    def __init__(self, n: float, domain: np.array, codomain: np.array, window_size: np.array):
        self.n = n
        self.domain = domain
        self.codomain = codomain
        self.window_size = window_size
        self.matrices = list(self._constructTransform())

    def _constructTransform(self):
        self.fs = int(1 / (self.domain[-1] - self.domain[-2]))
        time_window = np.linspace(-self.window_size / (2 * self.fs), self.window_size / (2 * self.fs),
                                  self.window_size, endpoint=False)
        self.dt = abs(time_window[-1] - time_window[-2])
        
        S = con.construct_S(self.n, self.codomain, time_window)
        T = con.construct_T(len(self.codomain))
        g = con.nGauss_wfunc(self.window_size, self.n)
        norm_g = integrate_vectors(g, g, self.dt) ** 2

        self.S = S
        self.T = T
        self.g = g
        self.norm_g = norm_g

        return [S, T, g]
    
    def _computeTransform(self, pixel_grid: PixelGrid) -> Signal:
        time_grid = np.matmul(
            self.S, np.matmul(self.T, pixel_grid.gridValue)
        )

        padded_time_grid = np.hstack(
            (
            np.zeros((self.window_size, self.window_size//2)), 
            time_grid, 
            np.zeros((self.window_size, self.window_size//2)))
        )

        diagonal_vectors = [
            np.array([padded_time_grid[j, j+t] for j in range(self.window_size)]) for t in range(len(self.domain))
        ]

        signal = list(
            map(lambda vector: integrate_vectors(vector, self.g, self.dt), diagonal_vectors)
        )

        return signal
    
    def transform(self, pixel_grid: PixelGrid) -> Signal:
        """ Inverse short time transform method. """
        signal = np.array(self._computeTransform(pixel_grid)) / self.norm_g
        return Signal(self.domain, signal)


class VVT(TransBC):
    def __init__(self, n_range: np.arange, domain: np.array, codomain: np.array, window_size: np.array):
        self.n_range = n_range
        self.domain = domain
        self.codomain = codomain
        self.window_size = window_size
        self.matrices = list(self._constructTransform())

    def _constructTransform(self):
        fs = int(1 / (self.domain[-1] - self.domain[-2]))
        time_window = np.linspace(-self.window_size / (2 * fs), self.window_size / (2 * fs),
                                  self.window_size, endpoint=False)
        matrices = []
        for n in tqdm(self.n_range, desc="Constructing Voxel Transform"):
            S = con.construct_S(n, time_window, self.codomain)
            T = con.construct_T(self.window_size)
            g = con.nGauss_wfunc(self.window_size, n) / integrate_vectors(
                con.nGauss_wfunc(self.window_size, n), con.nGauss_wfunc(self.window_size, n), 1 / fs
            )
            matrices.append((n, S, T, g))
        return matrices

    def transform(self, signal: Signal) -> VoxelGrid:
        voxel_array = self._computeTransform(signal)
        voxel_grid = VoxelGrid(self.domain, self.codomain, self.n_range, voxel_array)
        return voxel_grid

    def _computeTransform(self, signal: Signal) -> np.array:
        if len(signal.domain) == len(self.domain):
            voxel_grid = np.zeros((len(self.codomain), len(self.domain), len(self.n_range)), dtype=complex)
            for i in tqdm(range(len(self.n_range)), desc="Computing transform"):
                S = self.matrices[i][1]
                T = self.matrices[i][2]
                g = self.matrices[i][3]
                voxel_grid[:, :, i] = np.matmul(
                    S, np.matmul(
                        T, con.constructF(
                            signal.amplitude, self.window_size, g
                        )
                    )
                )
            return np.abs(voxel_grid)
        else:
            raise ValueError(f"Invalid signal input, domain must be same size as the transform domain. \n"
                             f"{len(signal.domain)} != {len(self.domain)}")


class VVT2:
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
            con.nGauss_wfunc(self.window_size, n) / integrate_vectors(
                con.nGauss_wfunc(self.window_size, n), con.nGauss_wfunc(self.window_size, n), 1 / self.fs
            )
        ) for n in self.n_range]

    def transform(self, signal: Signal) -> VoxelGrid:
        voxel_array = self._computeTransform(signal)
        voxel_grid = VoxelGrid(signal.domain, self.codomain, self.n_range, voxel_array)
        return voxel_grid

    def _computeTransform(self, signal: Signal) -> np.array:
        voxel_grid = np.zeros((len(self.codomain), len(signal.domain), len(self.n_range)), dtype=complex)
        for i in range(len(self.n_range)):
            S = self.matrices[i][1]
            T = self.matrices[i][2]
            g = self.matrices[i][3]
            voxel_grid[:, :, i] = np.matmul(
                S, np.matmul(
                    T, con.constructF(
                        signal.amplitude, self.window_size, g
                    )
                )
            )
        return np.abs(voxel_grid)

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


def integrate_vectors(v1: np.array, v2: np.array, step_size):
    """ This utilizes the dot product for efficient 
    integration given a constant step size. This
    is utilized by the ISTPT transform method."""
    return np.dot(v1, v2) * step_size

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dptcomp.dpfunc import s_n, c_n

    time = np.arange(-2, 2, 1/250)
    freq = np.arange(-4, 4, 1/250)
    n = 5

    forward = STPT(n, time, freq, 500)
    reverse = ISTPT(n, time, freq, 500)

    fig, ax = plt.subplots()

    amp = Signal(time, [np.exp(-t ** 2) * (np.sin(t ** 3) + .1 * np.cos(8 * t ** 5) + s_n(2, 3 * t)) for t in time])
    amp.populatePlot(ax)

    pix = forward.transform(amp)
    pix.plot()

    amp2 = reverse.transform(pix)
    amp2.plot(real=True)