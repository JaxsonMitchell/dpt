"""
This lays out the constructions of all the matrices described within the paper.
"""
import numpy as np
from comp.dpfunc import K


def construct_S(n: float, domain, codomain, inverse=False) -> np.array:
    domain = np.append(domain, domain[-1] * 2 - domain[-2])

    if not inverse:
        S = np.array(
            [
                [K(n, domain[i], codomain[j]) for i in range(len(domain))]
                for j in range(len(codomain))
            ]
        )
    else:
        S = np.array(
            [
                [K(n, domain[i], codomain[j]).conjugate() for i in range(len(domain))]
                for j in range(len(codomain))
            ]
        )
    return S


def construct_T(N: int) -> np.array:
    return np.array(
        [
            [-1 if i == j else 1 if i + 1 == j else 0 for i in range(N)]
            for j in range(N + 1)
        ]
    )


def constructIdenticalRows(amplitude: np.array, N: int):
    Ones = constructRowOnes(N)
    columnG = np.array(amplitude).reshape((len(amplitude), 1))
    return np.matmul(columnG, Ones)


def constructIdenticalCols(amplitude: np.array, N: int) -> np.array:
    Ones = constructColumnOnes(N)
    rowG = np.array(amplitude).reshape((1, len(amplitude)))
    return np.matmul(Ones, rowG)


def constructColumnOnes(N: int) -> np.array:
    return np.ones((N, 1))


def constructRowOnes(N: int) -> np.array:
    return np.ones((1, N))


def constructWJ(N: int, J: int) -> np.array:
    return np.array(
        [[1 if J > (j - i + J // 2) >= 0 else 0 for j in range(N)] for i in range(N)]
    )


def constructF(amplitude: np.array, J: int, g: np.array) -> np.array:
    """Computes the windows for f"""
    N = len(amplitude)

    return np.array(
        [
            [
                0
                if i + j - J // 2 < 0 or i + j - J // 2 > N - 1
                else amplitude[i + j - J // 2] * g[j]
                for i in range(N)
            ]
            for j in range(J)
        ]
    )


def nGauss_wfunc(N: int, n):
    """Automated window function for the vdpt."""
    # This initialization of the sigma value is so that it cuts off right when the window is done. It can be
    # changed depending on what you want from a window function.
    sigma = (1 / 3) / (2 * n - 1) ** (
        1 / (2 * n)
    )  # Places point of inflection close to the edge.

    return np.array(
        [
            np.exp(-((abs(x - 1) / (sigma * N)) ** (2 * n)) / (2 * n))
            for x in np.arange(-N // 2, N // 2)
        ]
    )


if __name__ == "__main__":
    domain = range(30)
    ampltiude = [2] + [1] * (len(domain) - 2) + [3]
    print(ampltiude)
    wj = constructF(ampltiude, 5)
    print(wj)
