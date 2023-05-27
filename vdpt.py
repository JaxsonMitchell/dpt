"""
Jaxson Mitchell

This code is the vector form of the DPT operation.
"""

from scipy.special import jv, gamma
import matplotlib.pyplot as plt
import numpy as np
from cmath import sqrt
import os


def g(n: float, x, y, threshold: float = .01): # g kernel defined in the paper.
    if abs(y) > threshold:
        return 1 / 2 * (np.sign(x) * sqrt(np.abs(x / y)) * jv(1 / (2.0 * n), (np.abs(x) * np.abs(y)) ** n / n) +
                        1j * np.sign(y) * ((2 * n) ** (1 / (2 * n)) / (np.abs(y) * gamma(1 - 1 / (2 * n)))
                                           - sqrt(abs(x / y)) * jv(- 1 / (2.0 * n), np.abs(x * y) ** n / n)))
    else:
        even = x / ((2 * n) ** (1 / (2 * n)) * gamma(1 + 1 / (2 * n)))
        odd = -1j * np.sign(y) * (2 * n) ** (1 / (2 * n) - 2) / (gamma(1 - 1 / (2 * n))) * \
              np.abs(x) ** (2 * n - 1) * np.abs(y) ** (2 * n + 1)
        return 1 / 2 * (even + odd)


def dpt(S, T, v):
    return np.matmul(S, np.matmul(T, v))


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


    return S, T


def construct_Transforms(n: float, domain: np.array, codomain: np.array, fs: int, inverse=False):
    N = len(domain)
    M = len(codomain)

    if not inverse:
        domain = np.append(domain, domain[-1] + 1 / fs)

        return construct_S(n, domain, codomain), construct_T(N)
    else:
        domain = np.append(domain, 2 * domain[-1] - domain[-2])

        return construct_S(n, domain, codomain, inverse=True), construct_T(N)


def construct_S(n: float, d, cod, inverse=False):
    d = np.append(d, 2 * d[-1] - d[-2])
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


if __name__ == '__main__':
    from time import process_time

    n_range = np.arange(1, 5, 1/10)
    fs_array = [128 * 2 ** i for i in range(4)]
    codomains = [range(64 * 2 ** i) for i in range(4)]
    domains = [1/16 * 2 ** i for i in range(6)]

    p = process_time()
    for n in n_range:
        print("-" * 10, n, "-" * 10)

        domain = np.arange(-2, 2, 1/4192)
        codomain = np.arange(0, 256, 1)

        save_Transform(n, domain, codomain)

    pf = process_time()"""

    fs = 4192
    n = 4.0
    domain = np.arange(-2, 2, 1/fs)
    Func = [np.sin((400 * t) ** n / n) for t in domain]
    codomain = np.arange(0, 256, 1)

    S, T = load_Transform(n, domain, codomain)
    output = dpt(S, T, Func)

    plt.plot(domain, Func)
    plt.show()

    plt.plot(codomain, output)
    plt.show()
    # print(f"The entire range took {pf - p} seconds!!! Wow that's a while!!!!")
    """fs = 100
    n = 3
    time = np.arange(-4, 4, 1 / fs)
    freq = np.arange(-5, 5, 1 / fs)
    Amp = [t ** (2 * n - 1) * np.exp(-t ** (2 * n) / (2 * n)) * np.sin(10 * t ** n)
           + np.exp(-t ** (2 * n) / (2 * n)) * np.sin(4 * t ** n) for t in time]

    S, T = save_Transform(n, time, freq)
    print(np.shape(S), np.shape(T))

    plt.plot(time, Amp)
    plt.show()

    print("Completed env. ")
    print("---")

    t2 = process_time()
    Func = dpt(S, T, Amp)
    t3 = process_time()

    print(f"Evaluation: {t3 - t2} seconds for a size {len(Amp)} signal")
    plt.plot(freq, np.abs(Func))
    plt.show()"""