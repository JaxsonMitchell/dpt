"""The paper corresponding to this has figures mainly generated from this file."""

import matplotlib.pyplot as plt
from dptcomp.dpfunc import s_n, c_n
import numpy as np
from dptcomp.sgn import Signal
from dpt.transform import DPT, VVT, STPT, ISTPT, VVT2

def figure_eigenfunctions():
    """ Figures for generating the eigenfunctions. """
    fs = 200
    time = np.arange(-3, 3, 1/fs)
    freq = np.arange(-3, 3, 1/fs)
    n = 4
    
    evenEig = Signal(time, np.exp(-time ** (2 * n) / (2 * n)))
    evenEig.labelSignal(fr"Even eigenfunction $\phi_0$")
    oddEig = Signal(time, time ** (2 * n - 1) * np.exp(-time ** (2 * n) / (2 * n)))
    oddEig.labelSignal(fr"Odd eigenfunction $\phi_1$")

    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Amplitude", fontsize=18)
    ax.set_title(fr"Eigenfunctions of the $\Phi_4$ transform", fontsize=18)

    evenEig.populatePlot(ax, plotReal=True, color="black")
    oddEig.populatePlot(ax, plotReal=True, color="red")

    plt.legend()
    plt.show()

    dpt = DPT(n, time, freq)
    transEvenEig = dpt.transform(evenEig)
    transEvenEig.labelSignal("Transformed Even eigenfunction")
    transOddEig = dpt.transform(oddEig)
    transOddEig.labelSignal("Transformed Odd eigenfunction")


    fig, ax = plt.subplots()
    ax.set_title(fr"Transformed eigenfunctions of the $\Phi_4$ transform", fontsize=18)
    ax.set_xlabel("4-Frequency", fontsize=18)
    ax.set_ylabel("Amplitude", fontsize=18)

    transEvenEig.populatePlot(ax, plotReal=True, color="black")
    transOddEig.populatePlot(ax, plotImaginary=True, color="red")

    plt.legend()
    plt.show()



def figure5_6_7_8():
    """Figures corresponding to chirps and their translation.
    This shows the translation variance of the transform when
    n does not equal 1."""
    time = np.arange(-3, 3, 1 / 100)
    freq = np.arange(-10, 10, 1 / 30)
    n = 3

    fig, ax = plt.subplots()
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Amplitude", fontsize=18)
    ax.set_title("Chirp signal of order 3", fontsize=18)

    amplitude = s_n(n, 2 * time) + 0.5 * c_n(n, 3 * time) + 0.05 * s_n(n, 5 * time)
    chirp3 = Signal(time, amplitude)
    chirp3.populatePlot(ax, plotReal=True)
    
    plt.show()
    
    translated_chirp = chirp3.translate(0.25)
    fig, ax = plt.subplots()
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Amplitude", fontsize=18)
    ax.set_title("Translated Chirp signal of order 3", fontsize=18)
    
    translated_chirp.populatePlot(ax, plotReal=True)

    plt.show()

    dpt_transform_1 = DPT(n, time, freq)
    dpt_transform_2 = DPT(n, time + 0.25, freq)

    norm_freq_3 = dpt_transform_1.transform(chirp3)
    trans_freq_3 = dpt_transform_2.transform(translated_chirp)

    norm_freq_3.labelSignal(
        fr"$\Phi_3$-transform of chirp signal", "3-frequency", "Amplitude"
    )
    trans_freq_3.labelSignal(
        fr"$\Phi_3$-transform of translated chirp signal", "3-frequency", "Amplitude"
    )

    fig, ax = plt.subplots()
    ax.set_xlabel("3-Frequency", fontsize=18)
    ax.set_ylabel("Amplitude", fontsize=18)
    ax.set_title("Transformed Chirp signal", fontsize=18)

    norm_freq_3.populatePlot(ax, plotAmplitude=True)
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel("3-Frequency", fontsize=18)
    ax.set_ylabel("Amplitude", fontsize=18)
    ax.set_title("Transformed Translated Chirp signal", fontsize=18)

    trans_freq_3.populatePlot(ax, plotAmplitude=True)
    plt.show()

    return None


def figures_9_10_11():
    """ Example short time transform """
    time = np.arange(0, 14, 1 / 400)
    freq = np.arange(0, 7, 1 / 400)
    n = 4
    window_size = 1600
    amplitude = np.exp(- (time - 2) ** n / n) * s_n(n, 1.8 * (time - 2)) + \
                np.exp(- (time - 8) ** (2 * n) / (2 * n)) * 1.2 * c_n(n, 2 * (time - 8)) + \
                np.exp(- (time - 12) ** (2 * n)) * (.27 * c_n(n, 4 * (time - 12)))
    
    
    fig, ax = plt.subplots()

    multichirp = Signal(time, amplitude)
    multichirp.populatePlot(ax, plotReal=True)
    
    ax.set_xlabel("Time", fontsize=16)
    ax.set_ylabel("Amplitude", fontsize=16)
    ax.set_title("Multi Chirp Signal", fontsize=20)

    plt.show()

    stpt = STPT(n, time, freq, window_size)
    # istpt = ISTPT(n, time, freq, window_size)
    pixel = stpt.transform(multichirp)
    pixel.setPlottingBehavior((100, 100), (12, 12), decimals=3, XY_font_labels=18)
    pixel.label("Short Time $\Phi_4$ Transform", "Time", "4-Frequency")
    pixel.plot()
    # new_sig = istpt.transform(pixel)

def figures12():
    fs = 200
    n_range = np.arange(1, 4, .05)
    time = np.arange(0, 10, 1 / fs)
    freq = np.arange(3, 7, .05)
    amp = (np.exp(-(time - 2) ** 6 / 6) * c_n(3, 4 * (time - 2)) +\
          np.exp(-(time - 7) ** 4 / 4) * s_n(2, 6 * (time - 7)) ) + np.random.normal(1, 3, len(time))

    multi_chirp = Signal(time, amp)
    fig, ax = plt.subplots(tight_layout=True)
    multi_chirp.populatePlot(ax, plotReal=True)
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Amplitude", fontsize=18)
    ax.set_title("Multi Chirp Order Signal", fontsize=20)
    plt.show()

    vvt = VVT(n_range, time, freq, 800)
    voxel_rep = vvt.transform(multi_chirp)
    """for time_window in [(2, 2.5), (7, 7.5),]:
        for thresh in [.75, .7, .65, .6, .5, .4, .3 ,.2]:
            print(thresh)
            voxel_rep.setPlottingBehavior(threshold=thresh, voxel_size=8, title=None)
            voxel_rep.setRegionOfInterest((3, 7), time_window, (1, 5))
            voxel_rep.plot()"""

    voxel_rep.makeVoxelVideo("testVoxel.mp4", 100, "Multi-Chirp Order Signal")


def figures13():
    fs = 200
    n_range = np.arange(2, 4, .1)
    time = np.arange(0, 10, 1 / fs)
    freq = np.arange(4, 6, 1 / 50)
    amp = np.exp(-(time - 2) ** 6 / 6) * c_n(3, 4 * (time - 2)) + np.exp(-(time - 7) ** 4 / 4) * s_n(2, 6 * (time - 7))

    multi_chirp = Signal(time, amp)
    multi_chirp.labelSignal("Multi Chirp Order Signal", "time", "amplitude")
    multi_chirp.plot()

    vvt = VVT2(n_range, fs, freq, 200)
    voxel_rep = vvt.transform(multi_chirp)

    """for time_window in [(1, 3), (2, 2.5), (6, 8), (7, 7.5), (0, 10)]:
        for thresh in [.99, .98, .97, .96, .95, .90, .85, .8, .75, .7, .65, .6]:
            print(thresh)
            voxel_rep.setPlottingBehavior(threshold=thresh, voxel_size=8, title=None)
            voxel_rep.setRegionOfInterest((3, 7), time_window, (1, 5))
            voxel_rep.plot()"""

    voxel_rep.plotGif("TimeProgression")
    voxel_rep.plotGif("FrequencyProgression", nFreq=True)
    voxel_rep.plotGif("ChirpOrderProgression", chirpOrder=True)


def figure14():
    """ Filter white noise """
    time = np.arange(0, 2, 1/500)
    freq = np.arange(0, 20, 1/30)
    n = 3

    dpt = DPT(n, time, freq)
    idpt = DPT(n, freq, time, inverse=True)
    
    sig = Signal(time, np.array([np.exp(-t ** (2 * n)) * s_n(3, 7 * t) for t in time]))
    sig.plot()
    
    tsig = dpt.transform(sig)
    tsig.plot(modulus=True)
    sig_new = idpt.transform(tsig)

    sig.populatePlot()
    sig_new.plot(real=True)


def figureChirp():
    """Figures corresponding to chirps and their translation.
    This shows the translation variance of the transform when
    n does not equal 1."""
    time = np.arange(-2.5, 2.5, 1 / 300)
    freq = np.arange(-10, 10, 1 / 300)
    n = 3

    fig, ax = plt.subplots()
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Amplitude", fontsize=18)
    ax.set_title("Chirp signal of order 3", fontsize=18)

    amplitude = np.exp(-time ** (2 * n) / 4) * c_n(n, 4 * time)
    chirp3 = Signal(time, amplitude)
    chirp3.populatePlot(ax, plotReal=True)
    
    plt.show()
    
    translated_chirp = chirp3.translate(0.25)
    
    dpt_transform_1 = DPT(n, time, freq)
    dpt_transform_2 = DPT(1, time, freq)

    norm_freq_3 = dpt_transform_1.transform(chirp3)
    norm_freq_1 = dpt_transform_2.transform(translated_chirp)

    norm_freq_3.labelSignal(
        fr"$\Phi_3$-transform of chirp signal", "3-frequency", "Amplitude"
    )
    norm_freq_3.labelSignal(
        fr"$\Phi_3$-transform of translated chirp signal", "3-frequency", "Amplitude"
    )

    fig, ax = plt.subplots()
    ax.set_xlabel("3-Frequency", fontsize=18)
    ax.set_ylabel("Amplitude", fontsize=18)
    ax.set_title("Transformed Chirp signal", fontsize=18)

    norm_freq_3.populatePlot(ax, plotAmplitude=True)
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel("Frequency", fontsize=18)
    ax.set_ylabel("Amplitude", fontsize=18)
    ax.set_title("Fourier Transform of Chirp signal", fontsize=18)

    norm_freq_1.populatePlot(ax, plotAmplitude=True)
    plt.show()

    return None


def figures_presentation():
    time = np.arange(0, 8, 1/4096)
    n1 = 1
    n2 = 2
    n3 = 3
    plt.plot(time, s_n(n1, 1 * time), label="Chirp Order n = 1")
    plt.plot(time, s_n(n2, 1 * time), label="Chirp Order n = 2")
    plt.plot(time, s_n(n3, 1 * time), label="Chirp Order n = 3")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.suptitle("Plots of s_n(t)")
    plt.title("Chirp Orders n = 1, 2, 3")
    plt.grid()
    plt.tight_layout()
    plt.show()

    time = np.arange(0, 30, 1/4096)
    w1 = .5
    w2 = 1
    plt.plot(time, c_n(1.2, w1 * time), label=fr"Chirp Frequency $\omega$ = .5", color="black")
    plt.plot(time, c_n(1.2, w2 * time), label=fr"Chirp Frequency $\omega$ = 1", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.suptitle("Plots of c_n(t)")
    plt.title(fr"n-Frequencies $\omega$ = .5, 1")
    plt.grid()
    plt.tight_layout()
    plt.show()

    time = np.arange(-10, 10, 1/4096)
    t01 = 0
    t02 = 3
    plt.plot(time, s_n(2, time - t01), label="t = 0")
    plt.plot(time, s_n(2, time - t02), label="t = 3")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.suptitle("Plots of s_n(t)")
    plt.title(fr"Centered Times: t = 0, 3")
    plt.grid()
    plt.tight_layout()
    plt.show()


#figure_eigenfunctions()
#figure5_6_7_8()
#figures_9_10_11()
figures12()
#figureChirp()
#figures_presentation()