"""
This is code for the Discrete Wavelet Phi Transform. It contains the following within this file:
- The discrete phi-transform computed through the wavelet approximation
- The lower functions used in the computation of the discrete phi-transform
- The inverse discrete phi-transform.
- Examples of utilizing these transforms on python.

Created by Jaxson Mitchell
"""

from scipy.special import jv, gamma
from scipy.signal import convolve
import numpy as np
from cmath import sqrt


# These functions are the wavelet decomposition of the Phi-transform. There are a few issues regarding the taylor
# expansion of x = 0 and some casting to complex errors, but aside from that it works as intended.


def dpt(n: float, data: np.array, fs: float = None, domain: np.array = None, epsilon=.1,
        output_domain: np.array = None):
    """
    :param output_domain: This will give the output_domain of a function, it will basically do the same thing
    as the normal dpt but compute the function over a specific area.
    :param domain: Instead of using a sampling frequency, you can simply use the array of time. This method doesn't rely
    on starting at t=0.
    :param epsilon: Threshold value where a power series approximation is used to prevent errors from 0/0 errors from
    the function shown in the notes.
    :param fs: This is the sampling frequency of the signal that you are using. This is utilized when generating the
    frequency values of the transform.
    :param n: The n-value of the transform, n = 1 is the Fourier transform and the larger n means it will check for
    more oscillatory behavior.
    :param data: This is the signal that is sampled at an equal time step. I wanted to have it be similar to the
    Fourier Transform in the sense that the discrete Fourier Transform does not need the time steps or the sampling
    frequency.
    :return: This returns the discrete phi-transform computed through the wavelet method. The frequencies that it is
    computed at will be the same as the Fourier Transforms [-fs/2, fs/2].
    """

    if output_domain is not None and domain is not None:  # Method 3
        # Has a desired outputted domain and domain. There is no sampling theorem for arbitrary domains, therefore
        # this is constructed for arbitrary data.
        return dpt_3(n=n, data=data, domain=domain, output_domain=output_domain, fs=fs)

    elif domain is None and fs is not None:  # Method 1
        # Have a sampling frequency with time assumed to start at t=0
        N = len(data)  # Obtains the length of the data
        return dpt_1(n=n, data=data, fs=fs, epsilon=epsilon, N=N)

    elif domain is not None and fs is not None and output_domain is None:  # Method 2
        # This is for if you have a domain of values that start at a nonzero value.
        freq_n_neg, freq_threshold_neg, freq_0, freq_threshold_pos, freq_n_pos \
            = dpt_sample(len(data), fs, n=n, threshold=epsilon)  # Gets the frequency values.
        return dpt_3(n=n, data=data, fs=fs, domain=domain, epsilon=epsilon, output_domain=np.concatenate((freq_n_neg,
                                                                                                          freq_threshold_neg,
                                                                                                          freq_0,
                                                                                                          freq_threshold_pos,
                                                                                                          freq_n_pos)))
    pass


def dpt_1(n: float, data: np.array, fs: float, epsilon: float, N: int, time_shift: float = 0) -> np.array:
    """
    :param time_shift: This is unnecessary in the majority of the calculations, but is used to simplify the calculations
    so that the dpt_2 method can just compute its transform by calling upon the dpt_1 method twice.
    :param N: The number of elements in the entire set of data.
    :param epsilon: Threshold value where a power series approximation is used to prevent errors from 0/0 errors from
    the function shown in the notes.
    :param fs: This is the sampling frequency of the signal that you are using. This is utilized when generating the
    frequency values of the transform.
    :param n: The n-value of the transform, n = 1 is the Fourier transform and the larger n means it will check for
    more oscillatory behavior.
    :param data: This is the signal that is sampled at an equal time step. I wanted to have it be similar to the
    Fourier Transform in the sense that the discrete Fourier Transform does not need the time steps or the sampling
    frequency.
    :return Returns the n-frequency content of the data.
    """
    freq_n_neg, freq_threshold_neg, freq_0, freq_threshold_pos, freq_n_pos \
        = dpt_sample(N, fs, n=n, threshold=epsilon)  # Gets the frequency values.

    freq_n_threshold = np.concatenate((freq_threshold_neg, freq_0, freq_threshold_pos))

    Func_threshold = np.zeros(len(freq_n_threshold), dtype=complex)
    Func_even_pos = np.zeros(len(freq_n_pos), dtype=complex)
    Func_odd_pos = np.zeros(len(freq_n_pos), dtype=complex)

    freq_n = np.concatenate(
        (freq_n_neg, freq_n_threshold, freq_n_pos)
    )  # Total frequency array

    data = np.append(data, 0)  # Adds a 0 at the end of the array to correspond f(t_N+1) data point which just
    # Simplifies calculations later on

    for num, d in enumerate(data):  # Iterates from 0, 1, 2, ..., N + 1  # Over N + 1 points
        """
        For each of the data points it will use equation 2.7s rewritten form to be able to compute the amplitude
        of the frequency at each point. It will do this by first having a variable c which corresponds to the difference
        f(t_k) - f(t_k-1). This is why the data was rewritten to have a 0 at the end.  This is because the first data 
        point would have the form f(t_1) - f(t_0) where f(t_0) = 0 and also f(t_N+1) = 0 can both be referred to 
        without it being necessary to write any if statements.

        Next, for each of the points it will calculate the complex value corresponding to the real frequency value. 
        It does this separately for the odd and even parts. For equation 2.7 it corresponds to the real and imaginary
        parts of the g(w, t) function. There are numerical errors from a divide by zero error, therefore about some
        threshold, there will be a separate function to calculate its frequency values.
        """

        # Variables used in the calculation
        c = data[num - 1] - data[num]
        t = num / fs + time_shift

        # The main computations of the Phi-transform.
        dpt_Even(n, c, t, freq_n_pos, Func_even_pos)
        dpt_Odd(n, c, t, freq_n_pos, Func_odd_pos)
        dpt_Threshold(n, c, t, freq_n_threshold, Func_threshold)

    # After calculating the positive frequency values, we can find the negative frequency values by
    # using the properties of the odd parts outputting an odd function and the even one outputting
    # an even function when inputting a wavelet.
    Func_even_neg = Func_even_pos[::-1]  # Flips the array around.  (Even function)
    Func_odd_neg = -1 * Func_odd_pos[::-1]  # Flips the array around and multiplies each element by -1 odd function

    # Sums the even and odd parts together
    Func_pos = Func_even_pos + Func_odd_pos
    Func_neg = Func_even_neg + Func_odd_neg

    Func = np.concatenate((Func_neg, Func_threshold, Func_pos))  # Creates the final output.

    return freq_n, Func


def dpt_2(n: float, data: np.array, fs: float, time: np.array, epsilon: float, inverse: bool = False) -> np.array:
    """
    :param inverse: This would compute the inverse transform instead of the normal transform
    :param epsilon: Threshold value where a power series approximation is used to prevent errors from 0/0 errors from
    the function shown in the notes.
    :param fs: This is the sampling frequency of the signal that you are using. This is utilized when generating the
    frequency values of the transform.
    :param n: The n-value of the transform, n = 1 is the Fourier transform and the larger n means it will check for
    more oscillatory behavior.
    :param data: This is the signal that is sampled at an equal time step. I wanted to have it be similar to the
    Fourier Transform in the sense that the discrete Fourier Transform does not need the time steps or the sampling
    frequency.
    :param time: This is the time array of the signal, so either the signal did not start at 0 or has negative values,
    both of these issues are resolved within this.
    :return Returns the n-frequency content of the data.
    """
    N = len(data)  # The length of the data that is used, assumed to be equal time steps.
    # Splits up the time array into two different sections for it to be utilized.
    negative_time = [t for t in time if t < 0]  # List comprehensions for time being negative
    positive_time = [t for t in time if t >= 0]  # List comprehensions for time being positive

    if len(negative_time) == 0:  # Only a translation positively in time
        time_shift = time[0]
        return dpt_1(n=n, data=data, fs=fs, epsilon=epsilon, N=N, time_shift=time_shift)
    elif len(positive_time) == 0:  # Only negative time values.
        time_shift = time[-1]  # An array flip will be used here.

        # I need to compute this still what would happen if t were to be flipped.
        freq, Func = dpt_1(n=n, data=data[::-1], fs=fs, epsilon=epsilon, N=N, time_shift=time_shift)
        return freq, -1 * Func[::-1]
    else:  # Both positive and negative time values
        time_shift = 1 / fs

        # Splits up the data about the positive and negative times
        negative_time_data = [data[i] for i in range(len(negative_time))]
        positive_time_data = [data[i] for i in range(len(negative_time), N)]

        # Positive time values
        freq, Func1 = dpt_1(n=n, data=positive_time_data, fs=fs, epsilon=.1, N=N)
        # Negative time values
        _, Func2 = dpt_1(n=n, data=negative_time_data[::-1], fs=fs, epsilon=.1, N=N, time_shift=time_shift)

        Func = Func1 + Func2[::-1]
        return freq, Func


def dpt_3(n: float, data: np.array, domain: np.array, output_domain: np.array, inverse: bool = False,
          epsilon: float = .1, fs: float = None) -> np.array:
    """
    :param fs: Sampling frequency of the signal, this doesn't have to be incorporated, but it is to ensure that
    this method is the same as the dpt_1 method which is the most optimized method, in reality without this, it is
    still able to extract chirps, this is only used for the t_k+1 term.
    :param n: The n-value of the transform, n = 1 is the Fourier transform and the larger n means it will check for
    more oscillatory behavior.
    :param data: This is the values given at each point in the domain.
    :param domain: This is the domain of the data itself. It can be arbitrary as long as its real.
    :param output_domain: This is the output domain, this is arbitrary as long as its real
    :param inverse: This boolean value describes whether the function will take the normal transform or its inverse.
    :param epsilon: This is a threshold value at which point a power series approximation will be used for the output
    :return: This returns the output domain as well as the outputted function given by the transform equation.
    """

    Negative_output_domain = [y for y in output_domain if y < -1 * epsilon]
    Positive_output_domain = [y for y in output_domain if y > epsilon]
    NonThreshold_output_domain = np.concatenate((Negative_output_domain, Positive_output_domain))
    Threshold_output_domain = [y for y in output_domain if -1 * epsilon <= y <= epsilon]

    # Non threshold output values
    Func_even_NT = np.zeros(len(NonThreshold_output_domain), dtype=complex)
    Func_odd_NT = np.zeros(len(NonThreshold_output_domain), dtype=complex)

    # Threshold output values
    Func_Threshold = np.zeros(len(Threshold_output_domain), dtype=complex)

    if fs is not None:
        data = np.append(data, 0)
        domain = np.append(domain, domain[-1] + 1 / fs)

    # Now for the even case, the positive and negative data can be split up and computed.
    for num, d in enumerate(data):  # Computes the positive domain data first
        """
        For each of the data points it will use equation 2.7s rewritten form to be able to compute the amplitude
        of the frequency at each point. It will do this by first having a variable c which corresponds to the difference
        f(t_k) - f(t_k-1). This is why the data was rewritten to have a 0 at the end.  This is because the first data 
        point would have the form f(t_1) - f(t_0) where f(t_0) = 0 and also f(t_N+1) = 0 can both be referred to 
        without it being necessary to write any if statements.

        Next, for each of the points it will calculate the complex value corresponding to the real frequency value. 
        It does this separately for the odd and even parts. For equation 2.7 it corresponds to the real and imaginary
        parts of the g(w, t) function. There are numerical errors from a divide by zero error, therefore about some
        threshold, there will be a separate function to calculate its frequency values.
        """
        c = data[num - 1] - data[num]  # f(t_k-1) - f(t_k)
        x = domain[num]  # Finds the corresponding domain number

        dpt_Even(n, c, x, NonThreshold_output_domain, Func_even_NT)  # Computes both negative and positive parts
        dpt_Odd(n, c, x, NonThreshold_output_domain, Func_odd_NT, inverse=inverse)
        dpt_Threshold(n, c, x, Threshold_output_domain, Func_Threshold, inverse=inverse)

    # Adding the two functions together
    Func_NT = Func_even_NT + Func_odd_NT

    # Finally, it will return the output domain and its corresponding output data.
    Func_Neg = Func_NT[0:len(Negative_output_domain)]
    Func_Pos = Func_NT[len(Negative_output_domain)::]

    Func = np.concatenate((Func_Neg, Func_Threshold, Func_Pos))
    return output_domain, Func


def dpt_sample(N: int, fs: float, n: float = 1, threshold=None, reverse: bool = False) -> np.array:
    """
    :param reverse: Reversing the sampling theorem to get a signal's time array.
    :param threshold: The threshold is a frequency value at which point
    :param fs: Sampling Frequency (Hz) of a * signal. Unlike the DFT, since we are using wavelets to approximate the
    signal instead of the sums, we have to have the sampling rate within this sample function.
    :param N: The N points from the inputted data
    :param n: This is the n-value of the transform, the Maximum frequency without aliasing depends on that value.
    :return: It will return the list of values used, this function is an approximation of a generalized sampling theorem
    that gets talked about in my notes
    """
    if threshold is None:
        Max_n_freq = n * fs ** (1 / n) / 2  # This value was calculated
        negative_freq = np.linspace(-Max_n_freq, 0, int(N / 2), endpoint=False)  # List of negative frequency values
        return negative_freq, [0], -1 * negative_freq[::-1]
    else:
        Max_n_freq = n * fs ** (1 / n) / 2  # Normal Shannon Sampling theorem value
        negative_freq = np.linspace(-Max_n_freq, 0, int(N / 2), endpoint=False)  # List of negative frequency values

        # Now that there is a threshold value, you want to be able to separate all the values. The for loop
        # iterates From the values of frequency closest to zero and then goes out from there. Each value in these
        # new arrays will either be all below or above the threshold value. When a frequency value is within the
        # threshold, it will be computed through a taylor expansion in a few terms.
        threshold_freq = negative_freq[negative_freq >= -1 * threshold]
        negative_freq = negative_freq[negative_freq < -1 * threshold]
        return negative_freq, threshold_freq, [0], -1 * threshold_freq[::-1], -1 * negative_freq[::-1]


def dpt_Even(n: float, c: float, t_k: float, freq: np.array, Func: np.array) -> None:
    """
    :param n: The n-value of the transform, n = 1 is the Fourier transform, and the larger n means it will check for
    more oscillatory behavior.
    :param c: f(t_k) - f(t_k-1) given by equation 2.7. Basically it's what is multiplying the output of the function.
    It's basically scalar multiple next to the output of the function.
    :param t_k: The time at which f(t_k) was sampled. This goes into the bessel function.
    :param freq: This is the frequency array that it will use to append over and utilize its frequency values.
    :param Func: This is the output of the function. It will take this array and then change the array.
    :return: It returns nothing, it just mutates the Func array.
    """
    if t_k > 0:
        for num, y in enumerate(freq):  # All the frequency values in the array as well as their positions.
            # This only takes positive frequency arrays, so there is no need for np.abs()
            Func[num] += c / 2 * (PnE(n, t_k, y))  # Even part of 2.7
    else:
        for num, y in enumerate(freq):  # All the frequency values in the array as well as their positions.
            # This only takes positive frequency arrays, so there is no need for np.abs()
            Func[num] -= c / 2 * (PnE(n, t_k, y))  # Even part of 2.7
    return None


def dpt_Odd(n: float, c: float, t_k: float, freq: np.array, Func: np.array, inverse: bool = False) -> None:
    """
    :param inverse: This is where, instead of using the normal PnO, you obtain it's conjugate which is just a flip
    by -1.
    :param n: The n-value of the transform, n = 1 is the Fourier transform, and the larger n means it will check for
    more oscillatory behavior.
    :param c: f(t_k) - f(t_k-1) given by equation 2.7. Basically it's what is multiplying the output of the function.
    It's basically scalar multiple next to the output of the function.
    :param t_k: The time at which f(t_k) was sampled. This goes into the bessel function.
    :param freq: This is the frequency array that it will use to append over and utilize its frequency values. It can
    also be the time values if taking the inverse transform.
    :param Func: This is the output of the function. It will take this array and then change the array.
    :return: It returns nothing, it just mutates the Func array.
    """
    if not inverse:
        for num, y in enumerate(freq):  # All the frequency values in the array as well as their positions.
            Func[num] += c / 2 * (PnO(n, t_k, y))  # Odd part of 2.7
        return None


def dpt_Threshold(n: float, c: float, t_k: float, freq: np.array, Func: np.array, inverse: bool = False) -> None:
    """
    :param inverse:
    :param n: The n-value of the transform, n = 1 is the Fourier transform and the larger n means it will check for
    more oscillatory behavior.
    :param c: f(t_k) - f(t_k-1) given by equation 2.7. Basically it's what is multiplying the output of the function.
    It's basically scalar multiple next to the output of the function.
    :param t_k: The time at which f(t_k) was sampled. This goes into the bessel function.
    :param freq: This is the frequency array that it will use to append over and utilize its frequency values.
    :param Func: This is the output of the function. It will take this array and then change the array.
    :return: It returns nothing, it just mutates the Func array.
    """
    for num, y in enumerate(freq):
        Func[num] += c / 2 * PnThreshold(n, t_k, y, inverse=inverse)
    return None


def PnThreshold(n: float, x: float, y: float, inverse: bool = False) -> complex:
    """
    :param n: The n-value of the transform, n = 1 is the Fourier transform, and the larger n means it will check for
    more oscillatory behavior.
    :param x: The domain value or time that the signal was sampled at.
    :param y: The frequency value that you are looking at.
    :param inverse: If it is an inverse or not.
    :return: This is the threshold function given by a taylor series expansion of the answer of solely for k = 0 on both
    parts
    """
    even = x / ((2 * n) ** (1 / (2 * n)) * gamma(1 + 1 / (2 * n)))
    odd = -1j * np.sign(y) * (2 * n) ** (1 / (2 * n) - 2) / (gamma(1 - 1 / (2 * n))) * \
          np.abs(x) ** (2 * n - 1) * np.abs(y) ** (2 * n + 1)
    return even + odd


def PnE(n: float, x: float, y: float) -> complex:
    """
    :param n: The n-value of the transform, n = 1 is the Fourier transform, and the larger n means it will check for
    more oscillatory behavior.
    :param x: The domain value or time that the signal was sampled at.
    :param y: The frequency value that you are looking at.
    :return: Returns the function given by the even part of 2.7
    """
    if x == 0:
        return 0 + 0j
    else:
        return sqrt(np.abs(x / y)) * jv(1 / (2 * n), (np.abs(x) * np.abs(y)) ** n / n)


def PnO(n: float, x: float, y: float) -> complex:
    """
    :param n: The n-value of the transform, n = 1 is the Fourier transform and the larger n means it will check for
    more oscillatory behavior.
    :param x: The domain value or time that the signal was sampled at.
    :param y: The frequency value that you are looking at. Since the frequency value is always positive for the
    ones we are looking at, we don't need the sgn(y) function.
    :return: Returns the function given by the even part of 2.7
    """

    if x == 0:
        return 0 + 0j
    else:
        return 1j * np.sign(y) * ((2 * n) ** (1 / (2 * n)) / (np.abs(y) * gamma(1 - 1 / (2 * n)))
                                  - sqrt(abs(x / y)) * jv(- 1 / (2.0 * n), np.abs(x * y) ** n / n))


def idpt(n: float, data: np.array, frequency: np.array, epsilon=.1, Negative_time: bool = False):
    """
    :param Negative_time: This is a boolean value that will determine whether the output will have negative time
    centered about t = 0 or simply a signal starting at t = 0.
    :param n: n-value of the inverse phi transform. It will transform the n-frequency back into its domain value (time)
    :param data: The data is the Amplitude of the frequency content which you will enter.
    :param frequency: The domain is the frequency data which corresponds to the data.
    :param epsilon: Threshold value of the inverse phi transform. This is when the taylor approximation will occur.
    :return: Returns a time array and the corresponding amplitude of the signal.
    """
    if not Negative_time:  # If there is no negative time signal starting at t = 0.
        threshold_time, pos_time = IDPT_Sample(n=n, frequency=frequency, threshold=epsilon)
        total_time = np.concatenate((threshold_time, pos_time))
    else:  # Creates the three arrays of time.
        neg_time, threshold_time, pos_time = IDPT_Sample(n=n, frequency=frequency, Negative_time=True,
                                                         threshold=epsilon)

        total_time = np.concatenate((neg_time, threshold_time, pos_time))

    Func_positive_even = np.zeros(len(pos_time), dtype=complex)
    Func_threshold = np.zeros(len(threshold_time), dtype=complex)
    Func_positive_odd = np.zeros(len(pos_time), dtype=complex)

    data = np.append(data, 0)  # Make it period corresponds to f(w_N + 1), f(w_0) = 0
    frequency = np.append(frequency, frequency[-1] + (frequency[-1] - frequency[-2]))

    for num, d in enumerate(data):
        c = data[num - 1] - d  # f(t_k-1) - f(t_k)
        w = frequency[num]  # Finds the corresponding domain number

        IDPT_Even(n, c, w, pos_time, Func_positive_even)
        IDPT_Odd(n, c, w, pos_time, Func_positive_odd)
        IDPT_Threshold(n, c, w, threshold_time, Func_threshold)

    if Negative_time:  # Gives the negative Time
        Func_negative_even = Func_positive_even[::-1]
        Func_negative_odd = -1 * Func_positive_odd[::-1]

        Func_pos_NT = Func_positive_even + Func_positive_odd
        Func_neg_NT = Func_negative_even + Func_negative_odd

        Func_tot = np.concatenate((Func_neg_NT, Func_threshold, Func_pos_NT))
    else:
        Func_NT = 2 * Func_positive_even + 2 * Func_positive_odd  # Gives the total signal thingy.
        Func_tot = np.concatenate((2 * Func_threshold, Func_NT))

    return total_time, Func_tot


def IDPT_Sample(n: float, frequency: np.array, Negative_time: bool = False, threshold=.01):
    """
    :param threshold: The threshold value specified, so that there values within it can be calculated using the
    taylor approximation of the output.
    :param Negative_time: Boolean value that lets the user specify whether the signal is centered about 0 or starts at 0
    :param n: n-value of the frequency domain that you will be inputting.
    :param frequency: Frequency domain that is supposedly equally spaced.
    :return: Returns the needed time values.
    """
    max_freq = frequency[-1]  # Maximum frequency of the data
    fs = (2 * max_freq / n) ** n

    if Negative_time:  # If there is negative time in the signal's domain.
        # Entire domain of signal.
        time = np.linspace(-1 / (2 * fs) * int(len(frequency / 2) - 1), 1 / (2 * fs) * int(len(frequency / 2) - 1),
                           len(frequency) - 1)

        threshold_time = [t for t in time if np.abs(t) <= threshold]
        pos_time = [t for t in time if t > threshold]
        neg_time = [t for t in time if t < -1 * threshold]

        return neg_time, threshold_time, pos_time

    else:  # If there isn't negative time in the signal's domain.
        # Entire domain of signal.
        time = np.linspace(0, 1 / fs * (len(frequency) - 2), len(frequency) - 1)

        # Splits it up into two components to minimize the numerical error about t = 0.
        threshold_time = [t for t in time if t <= threshold]
        NT_time = [t for t in time if t > threshold]
        return threshold_time, NT_time


def IDPT_Even(n: float, c: float, w_k: float, time: np.array, Func) -> None:
    """
    :param n: n-value of the transform. Higher n corresponds to more oscillatory data.
    :param c: c is from the equation for the inverse Phi transform f(w_k-1) - f(w_k)
    :param w_k: This is the frequency value of the corresponding even part of g(t, w_k).
    :param time: array of values that will be iterated over to get the signal.
    :param Func: Amplitude corresponding to each time value.
    :return: It mutates the Func array, so it doesn't return anything.
    """

    for num, t in enumerate(time):  # Because of the construction of IDPT, it only takes positive time and w_k values.
        Func[num] += c / 2 * (IPnE(n, t, w_k))  # Even part of 2.7
    return None


def IDPT_Odd(n: float, c: float, w_k: float, time: np.array, Func) -> None:
    """
    :param n: n-value of the transform. Higher n corresponds to more oscillatory data.
    :param c: c is from the equation for the inverse Phi transform f(w_k-1) - f(w_k)
    :param w_k: This is the frequency value of the corresponding even part of g(t, w_k).
    :param time: array of values that will be iterated over to get the signal.
    :param Func: Amplitude corresponding to each time value.
    :return: It mutates the Func array, so it doesn't return anything.
    """

    for num, t in enumerate(time):  # Because of the construction of IDPT, it only takes positive time and w_k values.
        Func[num] -= c / 2 * (IPnO(n, t, w_k))  # Even part of 2.7
    return None


def IDPT_Threshold(n: float, c: float, w_k: float, time: np.array, Func: np.array) -> None:
    """
    :param n: n-value of the transform. Higher n corresponds to more oscillatory data.
    :param c: c is from the equation for the inverse Phi transform f(w_k-1) - f(w_k)
    :param w_k: This is the frequency value of the corresponding even part of g(t, w_k).
    :param time: array of values that will be iterated over to get the signal.
    :param Func: Amplitude corresponding to each time value.
    :return: It mutates the Func array, so it doesn't return anything.
    """

    for num, t in enumerate(time):
        Func[num] += c / 2 * IPnThreshold(n, t, w_k)
    return None


def IPnE(n, x, y):
    """
    even part of the g function defined in the Phi Transform notes.
    """
    if y == 0:
        return 0 + 0j
    elif y > 0:
        return sqrt(np.abs(y / x)) * jv(1 / (2 * n), (np.abs(x) * np.abs(y)) ** n / n)
    else:
        return -1 * sqrt(np.abs(y / x)) * jv(1 / (2 * n), (np.abs(x) * np.abs(y)) ** n / n)


def IPnO(n, x, y):
    """
    odd part of the g function defined in the Phi Transform Notes.
    """
    if y == 0:
        return 0 + 0j
    else:
        return -1j * sqrt(np.abs(y / x)) * np.sign(x) * jv(- 1 / (2 * n), (np.abs(x) * np.abs(y)) ** n / n)


def IPnThreshold(n, x, y):
    """
    Inverse threshold transform.
    """
    even = y / ((2 * n) ** (1 / (2 * n)) * gamma(1 + 1 / (2 * n)))
    odd = 1j * np.sign(x) * (2 * n) ** (1 / (2 * n) - 2) / (gamma(3 - 1 / (2 * n))) * \
          np.abs(x) ** (2 * n) * np.abs(y) ** (2 * n - 1)
    return even + odd


def Gaussian_Noise(N: int, mean: float = 0, std: float = 1):
    """
    :param N: Number of points randomly chosen.
    :param std: Standard deviation of the noise.
    :param mean: Mean of the noise
    :return: Returns an array of gaussian distribution about the mean with a deviation of std and N points.
    """
    return np.random.normal(loc=mean, scale=std, size=N)


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


def filter_noise(signal: np.array, frequency: np.array, n: float, cutoff: float = 0, boost: float = 1,
                 inflection_shift: float = 0.0, convolution_size: int = 0, J: int = 5):
    """
    :param J: Series DAF filter number
    :param convolution_size: Size of the convolution, optional parameter where if this is not inputted, no convolution
    would be done.
    :param inflection_shift: How far you want to shift over the inflection if u
    :param boost: optional parameter. Due to loss of signal energy, through this high-pass filter, you can boost
    up the amplitude of the filter so that the low frequency content gets cut out less.
    :param cutoff: optional parameter. It gives what frequency value you would want the point of inflection to occur.
    :param n: The n-value is the n value of the transform that you apply. You don't need it to necessarily be that,
    if instead you would want your filter to be a very quick cut off you could increase your n-value accordingly.
    :param frequency: This is the frequency content you obtained from the dpt.
    :param signal: Input signal, usually obtained from the dpt.
    :returns a signal with a high pass filter applied which corresponds to less noise.
    """
    if cutoff == 0:  # If the user didn't input an initial cutoff
        max_freq_index = np.argmax(signal)
        freq_max = frequency[max_freq_index]

        sigma = freq_max + inflection_shift  # Lets max point
        # but with a shift by some amount be the point of inflection, and it calculates the corresponding
        # sigma value. this is talked more in depth in my paper.
    else:
        sigma = 2 * ((2 * n * (J + 1) - 2) ** (1 / (2 * n))) / cutoff

    if convolution_size > 0:
        kernel = [1 / convolution_size for _ in range(convolution_size)]

        signal = np.convolve(signal, kernel, mode='same')

    DAF_poly = np.zeros(len(frequency))
    for j in range(J + 1):
        DAF_poly += ((sigma * frequency) ** (2 * n) / (2 * n)) ** j / (gamma(j + 1))
    DAF = DAF_poly * np.exp(-(np.abs(frequency * sigma) ** (2 * n)) / (2 * n))

    plt.plot(DAF)
    plt.show()
    high_pass_filter = DAF

    return signal * high_pass_filter


def step(domain, codomain):
    domain_new = []
    range_new = []

    for num in range(len(domain)):
        if num == 0 or num == len(domain):
            domain_new.append(domain[num])
            range_new.append(codomain[num])
        else:
            domain_new += [domain[num], domain[num]]
            range_new += [codomain[num - 1], codomain[num]]

    return domain_new, range_new


def c_n(n: float, x):
    """
    :param n: n value of the generalized cosine function
    :param x: Input given
    :return: Returns the c_n function as defined in the paper.
    """
    if type(x) == float or type(x) == int:  # List of x values
        return c_n_comp(n, x)

    else:  # List type.
        y = []
        for itm in x:
            y.append(c_n_comp(n, itm))

        return np.array(y)


def c_n_comp(n: float, x):
    """
    Used in the c_n function, should not be necessary to call
    """
    if x == 0:
        return 1 / ((2 * n) ** (1 / (2 * n)) * gamma(1 + 1 / (2 * n)))
    else:
        return 1 / 2 * np.abs(x) ** (n - 1 / 2) * jv(-1 + 1 / (2 * n), (np.abs(x) ** n / n))


def s_n(n: float, x):
    """
    :param n: n-value of the generalized sine term
    :param x: Input given
    :return: returns the s_n function for some n and x as defined in the paper.
    """
    return -1 / 2 * np.sign(x) * np.abs(x) ** (n - 1 / 2) * jv(1 - 1 / (2 * n), (np.abs(x) ** n / n))


if __name__ == '__main__':
    import matplotlib.pyplot as plt  # Necessary imports

    # Example 1
    fs = 400
    n = 3

    time = np.arange(-2.3, 2.3, 1/fs)
    Amp = np.cos(10 * time ** n) * np.exp(-time ** 4 / 3)

    plt.title("Chirp signal")
    plt.plot(time, Amp, label="Chirp order n = 3", color="black")
    plt.show()

    freq, Func = dpt(3, Amp, fs=fs, domain=time, output_domain=np.arange(-10, 10, .05))

    plt.title(r"Discrete $\Phi_3$ Transform of the signal")
    plt.plot(freq, np.real(Func), label=r"Fourier transform of a chirp $\Phi_1$", color="black")
    plt.xlabel("Frequency")
    plt.show()


    # Example 2
    fs = 200
    n = 4

    time = np.arange(-3, 3, 1 / fs)
    Phi1 = np.exp(-time ** (2 * n) / (2 * n)) * time ** (2 * n - 1)
    Phi0 = np.exp(-time ** (2 * n) / (2 * n))
    freq, Func1 = dpt(n, data=Phi0, fs=fs, domain=time)
    _, Func2 = dpt(n, data=Phi1, fs=fs, domain=time)

    plt.plot(time, Phi0, label=r"$\phi_0$")
    plt.plot(time, Phi1, label=r"$\phi_1$")
    plt.xlabel("time (s)")
    plt.title(r"Eigenfunctions of the $\Phi_4$ transform")
    plt.legend()
    plt.show()

    plt.plot(freq, np.real(Func1), label=r"Real transform of $\phi_0$")
    plt.plot(freq, np.imag(Func2), label=r"Imaginary transform of $\phi_1$")
    plt.xlabel("4-frequency")
    plt.title(r"Transformed eigenfunctions of the $\Phi_4$ transform")
    plt.legend()
    plt.show()


    # Example 3
    fs = 300
    n = 2

    time = np.arange(-2, 2, 1/fs)
    time1 = np.arange(-2, 2, 1/10)
    Signal = np.sin(2 * time) + np.cos(3 * time ** 2) + np.exp(-1 * time ** 2)
    Signal1 = np.sin(2 * time1) + np.cos(3 * time1 ** 2) + np.exp(-1 * time1 ** 2)
    time1, Signal1 = step(time1, Signal1)
    plt.plot(time, Signal, color="black", linewidth=1.3, label="Actual Signal")
    plt.plot(time1, Signal1, color="red", linewidth=1, label="Step Approximation (fs = 10)")
    plt.legend()
    plt.show()

    # Example 4
    n = 4
    fs = 300
    time = np.arange(-5, 5, 1/fs)
    plt.plot(time, s_n(n, time), label=r"$s_3(t)$", color="black")
    plt.plot(time, c_n(n, time), label=r"$c_3(t)$", color="teal")
    plt.legend()
    plt.title(r"c_n and s_n components of the $\Phi_3$ transform")
    plt.show()

    # Example 5
    fs = 600
    time = np.arange(-3, 3, 1 / fs)
    n = 3

    Amp_g = time ** (2 * n - 1) * np.exp(-time ** (2 * n) / (2 * n)) * np.cos(5 * time ** n) + \
            np.exp(-time ** (2 * n) / (8 * n)) * np.cos(11 * time ** n)
    Amp = Amp_g + Gaussian_Noise(len(time), std=0.2)
    plt.plot(time, Amp, linewidth=.5, label="Noisy signal")
    plt.plot(time, Amp_g, linewidth=.8, color='black', label="Original signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude of signal")
    plt.title("Pure Signal")
    plt.legend()
    plt.show()

    # Transform
    freq, Func = dpt(n=n, data=Amp, domain=time, fs=fs)
    plt.plot(freq, np.real(Func), label='Real', linewidth=.7)
    plt.plot(freq, np.imag(Func), label='Imaginary', linewidth=.7)
    plt.suptitle("Noisy data")
    plt.title(fr"$\left(\Phi_3 f\right)(\omega)$")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    Func = filter_noise(Func, frequency=freq, n=n, cutoff=(n * 11 ** (1 / n)), convolution_size=3)
    plt.plot(freq, np.real(Func), label="Real", linewidth=.7)
    plt.plot(freq, np.imag(Func), label="Imaginary", linewidth=.7)
    plt.title(f"filtered {n}-frequency content")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.show()

    time, Amp2 = idpt(n, Func, frequency=freq, Negative_time=True)

    plt.plot(time, np.real(Amp2), linewidth=.7, label='Noise Reduced')
    plt.plot(time, time ** (2 * n - 1) * np.exp(-time ** (2 * n) / (2 * n)) * np.cos(5 * time ** n) +
             np.exp(-time ** (2 * n) / (8 * n)) * np.cos(11 * time ** n), linewidth=.7, color='black', label="Original "
                                                                                                             "Function")
    plt.xlabel("Time")
    plt.ylabel("Amplitude of signal")
    plt.title("Original vs. Filtered signal")
    plt.legend()
    plt.show()

