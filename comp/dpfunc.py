from scipy.special import jv, gamma
import numpy as np
from cmath import sqrt


def K(n: float, x, y, threshold: float = .1) -> complex:  # kernel defined in the paper.
    """ This is the kernel for the phi transform. It is used to compute both the inverse and the normal transformation
    as defined in the paper.

    The function g is computed by integrating the step function chi_{[a,b]}(x), which is simply
    0 outside of a and b and 1 within a and b. It provides a good decomposition of a discrete signal into wavelets with
    a relatively simple decomposition and computation.

    Args:
        :param n: n value of the Phi transform. increasing this means you increase the order of a chirp
        :param x: The first argument within g, in the normal transform, it's the n-frequency.
        :param y: The second argument within g, in the normal transform, it's the time.
        :param threshold: Due to the use of bessel functions and the denominator of the function, when numerically
        computing the values for small y values, a taylor expansion of the first even and odd terms are needed.

    Returns:
        :return: Returns a complex value.
    """
    if inThresholdOrZero(x, y, threshold):
        k = computeFirstTaylorK(n, x, y)
        return k
    else:
        k = computeK(n, x, y)
        return k


def inThresholdOrZero(x: float, y: float, threshold: float) -> bool:
    return True if inThreshold(y, threshold) or Zero(x) else False


def inThreshold(y: float, threshold: float) -> bool:
    return True if abs(y) <= threshold else False


def Zero(x: float) -> bool:
    return True if x == 0 else False


def computeK(n: float, x: float, y: float) -> complex:
    return 1 / 2 * (np.sign(x) * sqrt(np.abs(x / y)) * jv(1 / (2.0 * n), (np.abs(x) * np.abs(y)) ** n / n) +
                    1j * np.sign(y) * ((2 * n) ** (1 / (2 * n)) / (np.abs(y) * gamma(1 - 1 / (2 * n)))
                                       - sqrt(abs(x / y)) * jv(- 1 / (2.0 * n), np.abs(x * y) ** n / n)))


def computeFirstTaylorK(n: float, x: float, y: float) -> complex:
    return first_cn_taylor(n, x) + first_sn_taylor(n, x, y)


def first_cn_taylor(n: float, x: float) -> float:
    return 1 / 2 * (x / ((2 * n) ** (1 / (2 * n)) * gamma(1 + 1 / (2 * n))))


def first_sn_taylor(n: float, x: float, y: float) -> complex:
    return - 1j / 2 * np.sign(y) * (2 * n) ** (1 / (2 * n)) / (gamma(3 - 1 / (2 * n))) * \
            np.abs(x) ** (2 * n) * np.abs(y) ** (2 * n - 1)

def nGauss(n: float, x: float, sigma: float = 1):
    return np.exp(- (x / sigma) ** (2 * n) / (2 * n))


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
        return 1 / (2 * (2 * n) ** (1 / (2 * n)) * gamma(1 + 1 / (2 * n)))
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
    import matplotlib.pyplot as plt
    from time import process_time

    time = np.arange(-100, 100, 1/100)

    t1 = process_time()
    siggy = s_n(1.3, time)
    t2 = process_time()
    ciggy = c_n(1.3, time)
    t3 = process_time()

    print(f"{t2 - t1}s : {t3 - t2}s")
    plt.plot(time, siggy)
    plt.plot(time, ciggy)
    plt.show()
