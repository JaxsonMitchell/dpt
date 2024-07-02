

from time import process_time
import numpy as np
import pickle
import colorednoise as cn
import numpy as np
import matplotlib.pyplot as plt


def save_array_to_file(array, filename):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(array, file)
        print(f"Array saved to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")


def load_array_from_file(filename):
    try:
        with open(filename, 'rb') as file:
            loaded_array = pickle.load(file)
            return loaded_array
    except Exception as e:
        print("Error loading array:", e)
        return None

def colored_noise(exponent, length, amplitude=1.0):
    return amplitude * cn.powerlaw_psd_gaussian(exponent, length)


def white_noise(length, amplitude=1.0):
    return colored_noise(0, length, amplitude)


def pink_noise(length, amplitude=1.0):
    return colored_noise(1, length, amplitude)


def brown_noise(length, amplitude=1.0):
    return colored_noise(2, length, amplitude)


def makePhiPSD(time: list, freq: list, n: float, num_samples: int, noise_function: callable = white_noise):
    """ Makes the Power Spectral Density under the DPT 
    
    time - domain of noise
    freq - domain of the PSD
    n - chirp order of the PSD
    num_samples - number of noise samples used.
    noise_function - optional argument, specifies the noise function used 
    as long as it can be passed with only a length argument

    output - A PSD.
    """
    from dpt import DPT, Signal
    from time import process_time

    t0 = process_time()
    dpt = DPT(n, time, freq)
    t1 = process_time()

    print(f"--- DPT {n} takes {t1 - t0} seconds ---")

    PSD = np.zeros(len(freq))

    t0 = process_time()
    for _ in range(num_samples):  # Iterates over the number of noise samples.
        noise_signal = Signal(time, noise_function(len(time)))
        noise_frqeuency = dpt.transform(noise_signal)
        PSD += np.abs(noise_frqeuency.amplitude) ** 2
    t1 = process_time()
    print(f"--- {num_samples} DPT computations takes {t1 - t0} seconds ---")
    return PSD


if __name__ == '__main__':
    n_range = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    dir_name = "white_noise_psd"

    for n in n_range:
        PSD = load_array_from_file(f"{dir_name}/{n}PSD_White.pkl")
        freq = np.arange(0, 100, 1/100)

        plt.plot(freq[15:], PSD[15:])
        plt.grid()
        plt.title(fr"White Noise PSD (n = {n})")
        plt.xlabel(fr"{n}-Frequency")
        plt.ylabel("PSD Amplitude")
        plt.show()