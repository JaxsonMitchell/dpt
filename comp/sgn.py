import numpy as np
import matplotlib.pyplot as plt


class Signal:
    # Initializing signal
    def __init__(self, domain: np.array, amplitude: np.array):  # Initialize a signal
        self._setTimeAmplitudeValues(domain, amplitude)
        self._initializeXYLabels()

    def _setTimeAmplitudeValues(self, domain, amplitude):
        self.domain = np.array(domain)
        self.amplitude = np.array(amplitude)
        self._validateTimeAmplitudeSize()

    def _validateTimeAmplitudeSize(self):
        assert len(self.domain) == len(self.amplitude)

    def _initializeXYLabels(self):
        self.label = ""
        self.xlabel = "Time"
        self.ylabel = "Amplitude"

    # noinspection PyAttributeOutsideInit
    def labelSignal(
        self, name: str, xlabel: str = "Time", ylabel: str = "Amplitude"
    ):  # Populate signal's label's
        self.label = name
        self.xlabel = xlabel
        self.ylabel = ylabel

    # Plotting Signal
    def plot(self, modulus: bool = False, real: bool = False):
        fig, ax = plt.subplots()  # Create new figure and axes objects
        if modulus:
            self.populatePlot(ax, plotAmplitude=True)
        elif real:
            self.populatePlot(ax, plotReal=True)
        else:
            self.populatePlot(ax)

        self.setPlotProperties(ax)
        plt.show()

    def setPlotProperties(self, ax):
        ax.set_title(self.label)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

    def populatePlot(self, ax, plotAmplitude: bool = False, plotReal: bool = False):
        if plotAmplitude:
            ax.plot(
                self.domain,
                np.abs(self.amplitude),
                label=self.label,
                color="black",
                lw=0.6,
            )
        elif plotReal:
            ax.plot(
                self.domain,
                np.real(self.amplitude),
                label=self.label,
                color="black",
                lw=0.6,
            )
        else:
            ax.plot(
                self.domain, self.amplitude, label=self.label, color="black", lw=0.6
            )

    def __add__(self, signal):
        if self.domain.all() == signal.domain.all():
            difference = self.amplitude + signal.amplitude
            return Signal(self.domain, difference)
        else:
            raise ValueError("Signal domains are not equal")

    def __sub__(self, signal):
        if self.domain.all() == signal.domain.all():
            difference = self.amplitude - signal.amplitude
            return Signal(self.domain, difference)
        else:
            raise ValueError("Signal domains are not equal")

    def __mul__(self, value):
        if isinstance(value, Signal):
            product = self.amplitude - value.amplitude
        else:
            product = value * self.amplitude
        return Signal(self.domain, product)

    # Plotting step approx
    def plotStep(self):
        stepArrays = self.makeStep()
        plt.plot(stepArrays[0], stepArrays[1])
        self.setPlotProperties()
        plt.show()

    def makeStep(self):
        stepArrays = self.initializeDomains()
        self.populateDomains(stepArrays)
        return stepArrays

    @staticmethod
    def initializeDomains():
        return [[], []]

    def populateDomains(self, stepArrays: list):
        for num in range(len(self.domain)):
            if isOnEdgeOfArray(num, self.domain):
                self.addEndPoint(stepArrays, num)
            else:
                self.addStep(stepArrays, num)

    def addEndPoint(self, stepArrays, num):
        stepArrays[0].append(self.domain[num])
        stepArrays[1].append(self.amplitude[num])

    def addStep(self, stepArrays, num):
        stepArrays[0] += [self.domain[num], self.domain[num]]
        stepArrays[1] += [self.amplitude[num - 1], self.amplitude[num]]

    # Computations
    def getSamplingFrequency(self) -> int:
        # Assumption that data is equally spaced, if not, what are you doing??
        fs = int(1 / (self.domain[-1] - self.domain[-2]))
        return fs

    def translate(self, a: float):
        return Signal(self.domain + a, self.amplitude + a)


def isOnEdgeOfArray(index: int, domain: np.array) -> bool:
    return True if index == 0 or index == len(domain) else False


if __name__ == "__main__":
    time = np.arange(-3, 3, 1 / 30)
    func = [np.sin(t) for t in time]
    sine = Signal(time, func)
    sine.plotStep()
    sine.plot()
