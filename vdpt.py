"""
Jaxson Mitchell

Vector form of the discrete Phi_n transform. This software is meant to be a more optimized approach then previous
versions due to the use of matrices, list comprehensions, and the ability to precompute the matrices for the transform.
"""

# Imports
from scipy.special import jv, gamma
import numpy as np
from cmath import sqrt
import os


# Computations
def g(n: float, x, y, threshold: float = .01) -> complex:  # g kernel defined in the paper.
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
    if abs(y) > threshold:
        return 1 / 2 * (np.sign(x) * sqrt(np.abs(x / y)) * jv(1 / (2.0 * n), (np.abs(x) * np.abs(y)) ** n / n) +
                        1j * np.sign(y) * ((2 * n) ** (1 / (2 * n)) / (np.abs(y) * gamma(1 - 1 / (2 * n)))
                                           - sqrt(abs(x / y)) * jv(- 1 / (2.0 * n), np.abs(x * y) ** n / n)))
    else:
        even = x / ((2 * n) ** (1 / (2 * n)) * gamma(1 + 1 / (2 * n)))
        odd = -1j * np.sign(y) * (2 * n) ** (1 / (2 * n) - 2) / (gamma(1 - 1 / (2 * n))) * \
            np.abs(x) ** (2 * n - 1) * np.abs(y) ** (2 * n + 1)
        return 1 / 2 * (even + odd)


def dpt_mult(S: np.array, T: np.array, v: np.array):
    """ This is the vector version of the discrete transform. It is derived within the paper, but basically there are
    two matrices, one S and one T. The matrix S corresponds to the kernel g and is populated entirely with those
    functions (conjugate of the function if it's the inverse). T corresponds to how the function is decomposed, as long
    as the decomposition is of a step function and nothing more complicated.

    Args:
        :param S: M x N + 1 matrix
        :param T: N + 1 x N matrix
        :param v: dimension N column vector corresponding to frequency or time data.

    Return
        :return: A vector of dimensionality M

    Notes:
        The computation of the output vector is much more efficient than before in the dpt.py module, so I would
        recommend using this module instead. The true power of this transform comes with the fact in needing to
        do a large number of the same transformation, let's say analyzing a population of similar signals, you would
        only need one S and one T, initialize it, and simply do the multiplication.

        The longest part of this transform is the construction of S, sadly there are no easy optimizations or symmetries
        that I would be leverage to make its construction faster that I know of.
    """
    return np.matmul(S, np.matmul(T, v))


def v_dpt(v: np.array, n: float = None, domain: np.array = None,
          codomain: np.array = None, inverse: bool = False, S: np.array = None, T: np.array = None,):
    """ This is the encompassing function of the transform. Its return values are based off of the inputs given. One
    case it will construct and return S and T as well as the output vector w in S(Tv) if it is given S, T, and v.
    Another option, is simply if its given the n value, time domain, frequency domain, and v it will construct
    S, T, apply S(Tv) and return all the values w, S, T.

    Args:
        :param v: N column vector

        Constructors of the matrices
        :param n: n value of the transform, n = 1 is the normal Fourier transform.
        :param domain: domain of the signal, if normal transform it'd be the time data like when it was sampled.
        :param codomain: codomain of the signal, in the normal transform it'd be the n-frequency.
        :param inverse: Boolean value corresponding to whether the transform is the inverse transform. It tells S
        whether to be conjugated or not.

        Transform
        :param S: M x N + 1 matrix
        :param T: N + 1 x N matrix

    Return
        if S and T were specified
            :return: S(Tv)
        else
            :return: S(Tv), S, T
    """

    if S is None or T is None:  # Construction and Multiplication, if only one matrix is given, it will recompute both.
        S, T = construct_Transforms(n, domain, codomain, inverse)
        print(np.shape(S), np.shape(T))
        w = dpt_mult(S, T, v)

        return w, S, T
    else:  # Matrix multiplication only
        w = dpt_mult(S, T, v)

        return w


# Constructions of Matrices
def construct_Transforms(n: float, domain: np.array, codomain: np.array, inverse=False):
    N = len(domain)

    if not inverse:
        domain = np.append(domain, 2 * domain[-1] - domain[-2])

        return construct_S(n, domain, codomain), construct_T(N)
    else:
        domain = np.append(domain, 2 * domain[-1] - domain[-2])

        return construct_S(n, domain, codomain, inverse=True), construct_T(N)


def construct_S(n: float, d, cod, inverse=False):
    if not inverse:
        S = np.array([[g(n, d[i], cod[j]) for i in range(len(d))] for j in range(len(cod))])
    else:
        S = np.array([[g(n, d[i], cod[j]).conjugate() for i in range(len(d))] for j in range(len(cod))])

    print("S construction complete. ")
    print("---")

    return S


def construct_T(N):
    # Uses a list comprehension since they are faster than for loops
    return np.array([[-1 if i == j else 1 if i + 1 == j else 0 for i in range(N)] for j in range(N + 1)])


# Storage and Loading of Matrices (Still needs to be fixed, it also takes up a large amount of storage)
def save_Transform_S(n: float, domain: np.array, codomain: np.array) -> np.array:
    folder_path = f'matrices/S/{round(n, 5)}'  # generates a path to folder
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    file_names = os.listdir(folder_path)

    # Parameters for the domain
    lb_d = round(domain[0], 8)
    ub_d = round(domain[-1], 8)
    N = len(domain)

    # Parameters for the codomain
    lb_cod = round(codomain[0], 8)
    ub_cod = round(codomain[-1], 8)
    M = len(codomain)

    S_file_name = f'{lb_d}_{ub_d}_{N}_{lb_cod}_{ub_cod}_{M}.npy'

    if S_file_name not in file_names:
        S = construct_S(n, d=domain, cod=codomain)
        np.save(f"{folder_path}/{S_file_name}", S)
    else:
        S = np.array(np.load(f"{folder_path}/{S_file_name}"))

    return S


def save_Transform_T(N: int) -> np.array:
    folder_path = 'matrices/T/size'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    file_names = os.listdir(folder_path)
    if f"T{N}.npy" not in file_names:
        T = construct_T(N)
        np.save(f'{folder_path}/T{N}.npy', T)
    else:
        T = np.array(np.load(f'{folder_path}/T{N}.npy'))

    return T


def load_Transform(n, domain, codomain) -> (np.array, np.array):
    # Domain parameters
    lb_d = round(domain[0], 8)
    ub_d = round(domain[-1], 8)
    N = len(domain)

    # Parameters for the codomain
    lb_cod = round(codomain[0], 8)
    ub_cod = round(codomain[-1], 8)
    M = len(codomain)

    path = f'matrices/S/{round(n, 5)}/{lb_d}_{ub_d}_{N}_{lb_cod}_{ub_cod}_{M}.npy'
    S = np.array(np.load(path))

    path = f'matrices/T/size/T{N}.npy'
    T = np.array(np.load(path))

    return S, T


def flush_Matrices() -> None:
    """
    Deletes all matrices. Saves room, and in case a different method needs to be implemented, this ensures there are
    no overlaps.
    """
    import os
    import shutil

    print("Flushing...")

    # Define the path to the directory to empty
    directory = 'matrices'

    # Delete all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

    os.rmdir('matrices')

    print("Flush complete!")
    return None


def save_Transform(n: float, domain, codomain) -> [np.array, np.array]:
    """
    Constructs and saves the matrices S and T and saves them in a directory to be accessed later on. The structure is
    as follows

    Folder
    -S (Varies on n, domain, codomain) -> n -> domain/codomain
    -T (The only parameter for T is Size) -> size

    If S exists, then it will save
    :param codomain: The codomain or frequency which is what we are transforming to find.
    :param domain: The domain of the signal
    :param n: The n-value of the Phi transform, which corresponds to the chirp value.
    :return: S and T, but it can be ignored if you simply want to load it later.
    """
    from time import process_time

    # First creates the folder structure
    if not os.path.exists('matrices'):
        print("Creating folder structure...")
        os.mkdir('matrices')
        os.mkdir('matrices/S')
        os.mkdir('matrices/T')

    # S transform
    print("Saving S...")
    t0 = process_time()
    S = save_Transform_S(n, domain, codomain)
    t1 = process_time()

    print(f"S constructed in {t1 - t0} seconds")

    # T transform
    print("Saving T...")
    t2 = process_time()
    T = save_Transform_T(len(domain))
    t3 = process_time()
    print(f"T constructed in {t3 - t2} seconds")

    print("Completed!")
    print("-" * 10)

    return S, T


if __name__ == '__main__':
    from time import process_time
    import matplotlib.pyplot as plt

    fs = 200
    n = 3
    time = np.arange(-2, 2, 1/fs)
    freq = np.arange(0, 5, 1/100)
    v_vec = [np.exp(-t ** (2 * n)) * np.sin(10 * t ** n) for t in time]
    plt.plot(time, v_vec)
    plt.show()

    w_vec, _, _ = v_dpt(v_vec, n, domain=time, codomain=freq)
    plt.plot(freq, np.abs(w_vec))
    plt.show()

    x_vec, _, _ = v_dpt(w_vec, n, domain=freq, codomain=time, inverse=True)
    plt.plot(time, np.real(x_vec))  # Put np.real for numerical errors leading to small complex values almost negligible
    plt.plot(time, v_vec)
    plt.show()
