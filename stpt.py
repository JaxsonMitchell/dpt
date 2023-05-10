"""
This is the code for the time-frequency representation of the Phi transform implemented on python.
This code will contain the following:
- The short time Phi transform for an n-value
- A 3-dimensional short time Phi transform for a range of n-values
- Plotting functions for these 3-dimensional transforms
- An inverse short time Phi transform for a single n-value.
- Examples of the utilization of each of these

Created by Jaxson Mitchell
"""

import dpt
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt


# From the implementation of a Discrete Phi-Transform, we can use this to create a time-frequency representation
# of the Phi Transform. Below is the following code utilized in the short time Phi transform as well as the
# necessary code for generating window functions.


def stpt(n: float, data: np.array, g: Callable, g_size: int, skip_size: int, fs: float, freq: np.array = None,
         time_shift: float = 0.0):
    """
    :param time_shift: If the signal is shifted off from a certain time, then this will take that into account in the
    outputted time array, but it will not change any of the given information about the STPT.
    :param freq: Optional parameter that is used in the 3d n-time-frequency representation of the computation to
    make the graphing easier to understand.
    :param fs: Sampling frequency of the data. It doesn't have too much use currently since I need to improve this
    code.
    :param n: The n-value of the transform, n = 1 is the Fourier transform, and the larger n means it will check for
    more oscillatory behavior.
    :param data: Amplitude of a 1 dimensional signal. This is the input signal f
    :param g: g is the specific window you want to use for the stpt.
    :param g_size: Given a window, what is the size that you want to use
    :param skip_size: How many samples you skip for each time you calculate the Phi Transform
    :return: 2d array, time, frequency content
    """
    window = g(g_size)  # Creates a window function that will be slid across, this is a list of g_size elements
    """
    The time is on the x-axis or column part of the time-frequency plane of the stpt.
        The time plane can be figured out by finding the number of windows you would have before sliding through the 
        entire function. Here we are assuming that it won't slide through a window if it goes past the size of the 
        signal. This mean if the signal x has M-samples with a window size given by N = g_size, and with a skip size
        of P elements you can determine how many times you need to do the Phi Transform. Now if you start past M-N
        elements in the signal, then the window would go outside the signal so we only include it within that amount.
        It starts at the 0th sample. Each time it hops to the Pth item. So the starting parts of each window are given 
        by the following:
        x[0], x[P], x[2P], ... where some integer k*P =< M - N. k would be the number of iterations needed except it
        doesn't include the initial condition. So then solving for k you would have k + 1 =< (M - N)//P + 1 which 
        when rounded down gives the necessary equality.
    The frequency is on the y-axis or row part of the time-frequency plane of the stpt.
        By the shannon-sampling theorem, the maximum frequency value that can be obtained is fs/2 which needs to be in 
        Hz and not seconds. There is no rigorous generalized sampling theorem yet, but the sampling theorem only 
        determines the frequency values. It goes up to fs/2 with a step size of 1/N so there would be N + 1 frequency
        values included. When having a generalized sampling theorem, hopefully something similar will apply, so the
        number of rows shouldn't change, but the thing that will change is the maximum frequency with respect to the
        Generalized Nyquist frequency
    """
    if freq is None:  # The usual one that uses the generalized sampling theorem to compute it.
        output_matrix = np.zeros((int(1 + g_size // 2), int((1 + len(data) - g_size) // skip_size)))
        time_array = np.zeros(int((1 + len(data) - g_size) // skip_size))

        # Iterates all the time steps that were generated from above.
        for i in range(output_matrix.shape[1]):
            # First, you have to get the window from the signal. From the above logic on time you can show that it
            # starts at the iteration i * skip_size and ends at (i + 1) * skip_size
            data_window = data[i * skip_size: (i * skip_size) + g_size]  # A small window of A

            # Multiplies each part of the window with its window function
            data_new_window = data_window

            time = np.linspace(-g_size / (2 * fs), g_size / (2 * fs), g_size, endpoint=False)

            # Take the fast fourier transform of this function multiplied by the window function and then take only
            # the positive and zero frequency values
            freq, Func = dpt.dpt(n, data=data_new_window, fs=fs, domain=time)
            Transformed_Window = np.abs(Func[-int(1 + g_size // 2):])

            output_matrix[:, i] = Transformed_Window
            freq = freq[-int(1 + g_size // 2):]

            time_array[i] = (i * skip_size + g_size / 2) / fs + time_shift

    else:  # This is assumed to be used solely for the 3d stpt, so the frequency values will be positive
        time_array = np.zeros(int((1 + len(data) - g_size) // skip_size))
        output_matrix = np.zeros((len(freq), int((1 + len(data) - g_size) // skip_size)))

        # Iterates over each time step.
        for i in range(output_matrix.shape[1]):
            data_window = data[i * skip_size: (i * skip_size) + g_size]

            data_new_window = data_window * window
            time_window = np.linspace(-g_size / (2 * fs), g_size / (2 * fs), g_size, endpoint=False)
            _, Func = dpt.dpt(n, data=data_new_window, domain=time_window, output_domain=freq)

            Transformed_Window = np.abs(Func)
            output_matrix[:, i] = Transformed_Window

            time_array[i] = (i * skip_size + g_size / 2) / fs + time_shift

    return output_matrix, time_array, freq


def stpt_plot(data: np.array, time: np.array = None, freq: np.array = None,
              adjust_ratio: float = 1, title: str = None, xtick_size: int = 10, ytick_size: int = 10,
              y_skips: int = 1, x_skips: int = 1) -> None:
    """
    :param x_skips: Number of skips on the x label
    :param y_skips: Number of skips on the y label
    :param ytick_size: Size of y ticks.
    :param xtick_size: Size of x ticks.
    :param data: 2d short time Phi transform data.
    :param time: time data obtained from the short time phi transform
    :param freq: frequency content obtained from the short time phi transform
    :param adjust_ratio: Adjusting the size between the x and y-axis to make it more legible.
    :param title: Optional title to put on plot
    :return: None, it creates the plot.
    """
    # Create the plot
    fig, ax = plt.subplots()
    im = ax.matshow(data)

    if time is not None and freq is not None:
        ax.set_xticks(np.arange(0, len(time), x_skips))
        ax.set_yticks(np.arange(0, len(freq), y_skips))
        ax.set_xticklabels(time[::x_skips], fontsize=10, rotation=90)
        ax.set_yticklabels(freq[::y_skips], fontsize=10)

    if time is not None and freq is None:
        ax.set_xticks(np.arange(0, len(time), x_skips))
        ax.set_xticklabels(time[::x_skips], fontsize=10, rotation=90)

    if freq is not None and time is None:
        ax.set_yticks(np.arange(0, len(freq), y_skips))
        ax.set_yticklabels(freq[::y_skips], fontsize=10)

    plt.xticks(fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.xlabel("time")
    plt.ylabel("freq")
    ax.set_aspect(adjust_ratio)
    plt.title(title)
    plt.show()


def stpt_3d(data: np.array, g: Callable, g_size: int, skip_size: int, fs: float,
            freq: np.array = None, n_range: np.array = None, input_time: np.array = None):
    """
    :param input_time: The inputted time array can be used to determine a shifted time value which will appear
    on the axes when you plot. If this parameter isn't passed, it will just start at 0.
    :param fs: Sampling frequency of the data.
    :param freq: This is the frequency values that it will be iterating over, it will stay constant through
    :param n_range: The n-value of the transform, n = 1 is the Fourier transform, and the larger n means it will check
    for more oscillatory behavior. Now it will iterate through a range of n values
    :param data: This is the data that it is iterating over.
    :param g: The window function used for each of the transforms, it can and should be n-value dependent.
    :param g_size: The window functions size
    :param skip_size: The amount skipped per each value.
    :return: A 3-dimensional array
    """

    axis_length = int((1 + len(data) - g_size) // skip_size)
    # Calculating the axis length for time, if no input given, this length will be used for both the n/freq arrays.

    if n_range is None:  # If an input array isn't given, it will automatically generate one.
        n_range = np.linspace(.5, 5, axis_length, endpoint=False)

    if freq is None:  # If a frequency array isn't given, it will automatically generate one.
        freq = np.linspace(0, 10, axis_length, endpoint=False)

    output_matrix_3d = np.zeros((len(freq), axis_length, len(n_range)))

    for num, n in enumerate(n_range):  # Iterates over each n value
        output_matrix_3d[:, :, num], time, freq = stpt(n=n, data=data, g=g, g_size=g_size, skip_size=skip_size, fs=fs,
                                                       freq=freq)

    return output_matrix_3d, time, freq, n_range


def nGauss(N: int, sigma: float = None, n: float = 1, func_type: str = "even"):
    """
    :param func_type: This shows whether you want the odd or even n-Gaussian function
    :param N: This is the number of elements you are windowing
    :param sigma: This is the deviation of the n-Gaussian
    :param n: This is the degree n of the transform used
    :return: Returns the n-Gauss window which can be applied on both frequency or time data as a filter. It can
    also be utilized as a window function for the n gaussian.
    """
    if sigma is None:
        # This initialization of the sigma value is so that it cuts off right when the window is done. It can be
        # changed depending on what you want from a window function.
        sigma = 1 / np.sqrt(2 * n)

    if func_type == "even":
        return np.array(
            [np.exp(-np.pi * (x / (sigma * N)) ** (2 * n) / (2 * n))
             for x in np.arange(-N // 2, N // 2)])
    elif func_type == "odd":  # Odd base eigenfunction
        return np.array(
            [(x / (sigma * N)) ** (2 * n - 1) * np.exp(-np.pi * (x / (sigma * N)) ** (2 * n) / (2 * n))
             for x in np.arange(-N // 2, N // 2)])


def max_in_array(data: np.array) -> (float, tuple):
    """
    :param data: 3d/2d array. Finds where the maximum within the array is.
    :return: Returns the maximum value and where in the array the maximum value is at.
    """

    Max_val = np.max(data)  # Finds the maximum within the array.
    Max_Index = np.where(data == Max_val)

    return Max_val, Max_Index


def normalize_array(data: np.array) -> np.array:
    """
    :param data: The array that will be normalized
    :return: A normalized array.
    """
    val_max, _ = max_in_array(data)  # Finds the maximum value so that normalization occurs
    data = data.astype(float)  # Converts the array to a float type so division doesn't mess up.
    data /= val_max  # Divides the entire array of data by the maximum value.

    return data


def threshold_values_in_array(data: np.array, threshold: float = .85) -> np.array:
    """
    :param threshold: Threshold percentage that returns all values that are 85% the maximum for the default.
    :param data: 3-dimensional data/2-dimensional data
    :return: Returns all points that are some threshold of the max value.
    """
    Normed_data = normalize_array(data)
    indices_list = np.where(Normed_data > threshold)
    if data.ndim == 3:  # If the data is three-dimensional
        indices_values = [data[indices_list[0][i], indices_list[1][i], indices_list[2][i]]
                          for i in range(len(indices_list[0]))]
    elif data.ndim == 2:  # 2 dimensional data
        indices_values = [data[indices_list[0][i], indices_list[1]] for i in range(len(indices_list[0]))]
    else:
        raise ValueError("data array must either be 2 or 3 dimensions.")
    return np.row_stack((indices_list[0], indices_list[1], indices_list[2], indices_values))


def stpt_3d_plot(data: np.array, time: np.array, freq: np.array, n_range: np.array, threshold: float = .5,
                 cmap: str = 'hot', s: int = 20, marker: str = 's', alpha: float = .8,
                 t_lims: tuple = None, freq_lims: tuple = None, n_lims: tuple = None, title: str = None) -> None:
    """
    :param title: Optional: Title of the figure
    :param n_lims: varies the n-range that you see.
    :param freq_lims: optional parameter varies the frequency range that you see.
    :param t_lims: optional parameter varies the time range that you see.
    :param data: 3-dimensional array
    :param time: time data
    :param freq: frequency data
    :param n_range: n-value range
    :param alpha: transparency of the 3d plot.
    :param marker: marker used on the 3d plot.
    :param cmap: cmap type
    :param s: size of the marker
    :param threshold: Threshold that takes out more or less values.
    """

    output_values = data
    maximum, _ = max_in_array(output_values)

    tt, ff, nn = np.meshgrid(time, freq, n_range, indexing='ij')
    # Creates the 3d map.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Assume `x`, `y`, `z` are 3D arrays of the same shape as `output_values`
    indices = np.argwhere(output_values > threshold * maximum)  # Get indices where condition is met
    values = output_values[indices[:, 0], indices[:, 1], indices[:, 2]]  # Extract values at those indices

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(freq[indices[:, 0]], time[indices[:, 1]], n_range[indices[:, 2]],
               c=values, marker=marker, alpha=alpha, cmap=cmap, s=s)
    ax.set_xlabel('n-frequency')
    ax.set_ylabel('time')
    ax.set_zlabel('n-value')

    if t_lims is not None and len(t_lims) == 2:
        ax.set_xlim(t_lims[0], t_lims[1])

    if freq_lims is not None and len(freq_lims) == 2:
        ax.set_ylim(freq_lims[0], freq_lims[1])

    if n_lims is not None and len(n_lims) == 2:
        ax.set_zlim(n_lims[0], n_lims[1])

    plt.show()
    return None


if __name__ == '__main__':
    # Gravitational Waveform maker
    # from pycbc.waveform import get_td_waveform
    import matplotlib.pyplot as plt

    # Example 1
    fs = 200  # Sampling frequency Number of points per second
    n = 3
    time = np.arange(0, 10, 1 / fs)
    frq1 = 4.27

    frq2 = 3
    Amp = 1.33 * np.exp(-(time - 3.5) ** (2 * n) / (2 * n)) * np.cos((frq1 * (time - 3.5)) ** n / n) + \
        np.exp(-(time - 8) ** (2 * n) / (2 * n)) * (np.sin((frq2 * (time - 8)) ** n / n))

    plt.title("Multi-chirp signal")
    plt.plot(time, Amp, label=f"Localized chirps around t=3.5, t=8", color="black", linewidth=.6)
    plt.xlabel("time (s)")
    plt.show()

    output, time, freq = stpt(n, Amp, fs=fs, g_size=400, skip_size=25, g=nGauss)
    tickvalues = [round(y, 3) for y in freq]
    stpt_plot(output, time, tickvalues, adjust_ratio=len(time)/len(tickvalues), ytick_size=8,
              title=r"Short-Time $\Phi_3$ Transform", x_skips=2, y_skips=6)

    fs = 100
    time = np.arange(0, 20, 1 / fs)
    amp = np.exp(-(time - 3) ** 2 / 2) * np.cos(6 * (time - 3)) + \
          np.exp(- (time - 10) ** 6 / 3) * np.sin((3 * (time - 10)) ** 3 / 3) * .6 \
          + .7 * np.exp(-(time - 16) ** 2 / .5) * np.cos((4 * (time - 16)) ** 4 / 4)

    plt.plot(time, amp, color="black", label="multi chirp")
    plt.title("multi-chirp signal")
    plt.plot(time, np.exp(-(time - 3) ** 2 / 2) * np.cos(6 * (time - 3)), label="n = 1, t = 3", linewidth=.7)
    plt.plot(time, np.exp(- (time - 10) ** 2 / 3) * np.sin((3 * (time - 10)) ** 3 / 3) * .6, label="n = 3, t = 10",
             linewidth=.7, color="black")
    plt.plot(time, .7 * np.exp(-(time - 16) ** 2 / .5) * np.cos((4 * (time - 16)) ** 4 / 4), label="n = 4, t = 16",
             linewidth=.7, color="green")
    plt.xlabel("time")
    plt.legend()
    plt.show()

    output_values, time, freq, n_val = stpt_3d(data=amp, g_size=200, g=np.hanning, skip_size=25, fs=fs,
                                               freq=np.arange(0, 20, .25), n_range=np.arange(1, 5, 1))
    stpt_3d_plot(data=output_values, time=time, freq=freq, n_range=n_val, threshold=.12)
