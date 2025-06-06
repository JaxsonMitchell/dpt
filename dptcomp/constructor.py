"""
Name: Jaxson Mitchell

This lays out the constructions of all the matrices described within the paper. It's a pretty good 
paper you should read it sometime :)
"""


import numpy as np
from dptcomp.dpfunc import K
from tqdm import tqdm


def construct_S(n: float, domain, codomain, inverse=False) -> np.array:
    domain = np.append(domain, domain[-1] * 2 - domain[-2])
    output_domain = tqdm(range(len(codomain)), desc=f"Constructing S Matrix (Chirp Order {n:.4f})")
    
    if not inverse:
        S = np.array(
            [
                [K(n, domain[i], codomain[j]) for i in range(len(domain))]
                for j in output_domain
            ]
        )
    else:
        S = np.array(
            [
                [K(n, domain[i], codomain[j]).conjugate() for i in range(len(domain))]
                for j in output_domain
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

def nGauss_wfunc(N: int, n, drop: float = .05):
    """Automated window function for the Voxel Transform."""
    domain = np.linspace(-(2 * n * np.log(1 / drop)) ** (1 / (2 * n)), (2 * n * np.log(1 / drop)) ** (1 / (2 * n)), N)
    window = np.exp(- abs(domain) ** (2 * n) / (2 * n))
    return 1 / np.sqrt(np.dot(window, window)) * window # Gotta normalize it homie.
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 1000
    T = construct_T(N)

    n        = 5/8
    domain   = np.arange(0, 1, 1/1024)
    codomain = np.arange(0, 1, 1/1024)
    S = construct_S(n, domain, codomain)

    n = 3 
    N = 1024
    plt.plot(nGauss_wfunc(N, n))
    plt.show() 

    n = .333 
    N = 1024
    plt.plot(nGauss_wfunc(N, n))
    plt.show()