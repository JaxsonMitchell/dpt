"""The paper corresponding to this has figures mainly generated from this file."""

import matplotlib.pyplot as plt
from comp.dpfunc import s_n, c_n
import numpy as np
from comp.sgn import Signal
from dpt.transform import DPT, VVT, STPT, ISTPT, VVT2


def figure5_6_7_8():
    """Figures corresponding to chirps and their translation.
    This shows the translation variance of the transform when
    n does not equal 1."""
    time = np.arange(-3, 3, 1 / 400)
    freq = np.arange(-10, 10, 1 / 30)
    n = 3

    amplitude = s_n(n, 2 * time) + 0.5 * c_n(n, 3 * time) + 0.05 * s_n(n, 5 * time)
    chirp3 = Signal(time, amplitude)
    chirp3.labelSignal("Chirp signal of order 3", "Time", "Amplitude")
    chirp3.plot()

    translated_chirp = chirp3.translate(0.25)
    translated_chirp.labelSignal(
        "Chirp signal translated .1 units", "Time", "Amplitude"
    )
    translated_chirp.plot()

    dpt_transform_1 = DPT(n, time, freq)
    dpt_transform_2 = DPT(n, time + 0.25, freq)

    norm_freq_3 = dpt_transform_1.transform(chirp3)
    trans_freq_3 = dpt_transform_2.transform(translated_chirp)

    norm_freq_3.labelSignal(
        r"$\Phi_3$-transform of chirp signal", "3-frequency", "Amplitude"
    )
    trans_freq_3.labelSignal(
        r"$\Phi_3$-transform of translated chirp signal", "3-frequency", "Amplitude"
    )

    norm_freq_3.plot(modulus=True)
    trans_freq_3.plot(modulus=True)

    return None


def figures_9_10_11():
    """ Example short time transform """
    time = np.arange(0, 14, 1 / 400)
    freq = np.arange(-7, 7, 1 / 200)
    n = 4
    window_size = 1600
    amplitude = np.exp(- (time - 2) ** n / n) * s_n(n, 1.8 * (time - 2)) + \
                np.exp(- (time - 8) ** (2 * n) / (2 * n)) * 1.2 * c_n(n, 2 * (time - 8)) + \
                np.exp(- (time - 12) ** (2 * n)) * (.27 * c_n(n, 4 * (time - 12)))
    multichirp = Signal(time, amplitude)
    multichirp.labelSignal("Multi Chirp Signal", "Time", "Amplitude")
    multichirp.plot()
    stpt = STPT(n, time, freq, window_size)
    istpt = ISTPT(n, time, freq, window_size)
    pixel = stpt.transform(multichirp)
    pixel.setPlottingBehavior((80, 40), (6, 6))
    pixel.label("Short Time $\Phi_4$ Transform", "Time", "4-Frequency")
    pixel.plot()
    new_sig = istpt.transform(pixel)
    new_sig.labelSignal("Inverse Short Time $\Phi_4$ Transform", "Time", "Amplitude")
    new_sig.plot(real=True)

def figures12():
    fs = 200
    n_range = np.arange(2, 4, .02)
    time = np.arange(0, 10, 1 / fs)
    freq = np.arange(3, 7, 1/ 100)
    amp = np.exp(-(time - 2) ** 6 / 6) * c_n(3, 4 * (time - 2)) + np.exp(-(time - 7) ** 4 / 4) * s_n(2, 6 * (time - 7))

    multi_chirp = Signal(time, amp)
    multi_chirp.labelSignal("Multi Chirp Order Signal", "time", "amplitude")
    multi_chirp.plot()

    vvt = VVT(n_range, time, freq, 800)
    voxel_rep = vvt.transform(multi_chirp)
    for time_window in [(1, 3), (2, 2.5), (6, 8), (7, 7.5), (0, 10)]:
        for thresh in [.99, .98, .97, .96, .95, .90, .85, .8, .75, .7, .65, .6]:
            print(thresh)
            voxel_rep.setPlottingBehavior(threshold=thresh, voxel_size=8, title=None)
            voxel_rep.setRegionOfInterest((3, 7), time_window, (1, 5))
            voxel_rep.plot()



def figures13():
    fs = 50
    n_range = np.arange(2, 4, .02)
    time = np.arange(2, 8, 1 / fs)
    freq = np.arange(4, 6, 1/ 100)
    amp = np.exp(-(time - 2) ** 6 / 6) * c_n(3, 4 * (time - 2)) + np.exp(-(time - 7) ** 4 / 4) * s_n(2, 6 * (time - 7))

    multi_chirp = Signal(time, amp)
    multi_chirp.labelSignal("Multi Chirp Order Signal", "time", "amplitude")
    multi_chirp.plot()

    vvt = VVT2(n_range, fs, freq, 200)
    voxel_rep = vvt.transform(multi_chirp)
    vvt.saveTransform("Testing.h5")

    for time_window in [(1, 3), (2, 2.5), (6, 8), (7, 7.5), (0, 10)]:
        for thresh in [.99, .98, .97, .96, .95, .90, .85, .8, .75, .7, .65, .6]:
            print(thresh)
            voxel_rep.setPlottingBehavior(threshold=thresh, voxel_size=8, title=None)
            voxel_rep.setRegionOfInterest((3, 7), time_window, (1, 5))
            voxel_rep.plot()

"""def generate_noise_psds():
    from dpt.transform import 
    from time import process_time
    import pickle
    time = np.arange(0, 1, 1 / 1_000)
    freq = np.arange(0, 100, 1/100)
    PSD = np.zeros(len(freq))
    n = 3
    N = 100_000
    t1 = process_time()
    dpt = DPT(n, time, freq)
    t2 = process_time()
    print(f"DPT size {len(freq) * len(time)} took {t2 - t1} seconds")
    for i in range(N):
        noise = generate_noise(len(time), 2)
        noisy_signal = Signal(time, noise)
        noisy_frequency = dpt.transform(noisy_signal)
        PSD += np.abs(noisy_frequency.amplitude) ** 2 / N
    t3 = process_time()
    print(f"Analysis of {N} signals took {t3 - t2} seconds")
    save_array_to_file(PSD, "PowerSpectralDensity_brown")
    save_array_to_file(freq, "Frequency")
    PowerSpectralDensity = Signal(freq, PSD)
    PowerSpectralDensity.labelSignal(r"PSD of white noise n = 3 $\sigma = 1$", "Time", "3-Frequency")
    PowerSpectralDensity.plot()
"""

if __name__ == '__main__':
    figures13()