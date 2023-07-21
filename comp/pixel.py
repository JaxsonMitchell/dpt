import numpy as np
import matplotlib.pyplot as plt


class PixelGrid:
    def __init__(
        self, time: np.array, frequency: np.array, gridValue: np.array, n: float = 1
    ):
        self._setTimeFreqAmplitudeValues(time, frequency, gridValue)
        self._initializePlottingBehavior()
        self.n = n
        self._setLabels()

    def _setTimeFreqAmplitudeValues(self, time, frequency, gridValue):
        self.time = time
        self.frequency = frequency
        self.gridValue = gridValue
        self._validateAmplitudeSize()

    def _setLabels(self):
        self.name = ""
        self.xlabel = "Time"
        self.ylabel = f"{self.n}-Frequency"

    def _validateAmplitudeSize(self):
        assert np.shape(self.gridValue)[1] == len(self.time)
        assert np.shape(self.gridValue)[0] == len(self.frequency)

    def _initializePlottingBehavior(self):
        self.__XY_skips = (1, 1)
        self.__XY_fontsize = (12, 12)
        self.decimals = 8

    # noinspection PyAttributeOutsideInit
    def setPlottingBehavior(
        self, XY_skips: tuple, XY_fontsize: tuple, decimals: int = 8
    ):
        self.__XY_skips = XY_skips
        self.__XY_fontsize = XY_fontsize
        self.decimals = decimals

    def label(self, name, xlabel, ylabel):
        self.name = name
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self):
        fig, ax, im = create2DPlot(np.abs(self.gridValue))

        set2DLabels(ax, self.gridValue, self.__XY_skips)
        setTimeFreqAxes(
            ax,
            (
                np.round(self.time, self.decimals),
                np.round(self.frequency, self.decimals),
            ),
            self.__XY_skips,
            self.__XY_fontsize,
            self.xlabel,
            self.ylabel,
            self.name,
        )
        ax.set_aspect(len(self.time) / (2 * len(self.frequency)))
        plt.show()


class VoxelGrid:
    def __init__(
        self,
        time: np.array,
        frequency: np.array,
        n_range: np.array,
        gridValue: np.array,
    ):
        self._setTimeFreqAmplitudeValues(time, frequency, n_range, gridValue)
        self._initializePlottingBehavior()

    def _setTimeFreqAmplitudeValues(self, time, frequency, n_range, gridValue):
        self.time = time
        self.frequency = frequency
        self.n_range = n_range
        self.gridValue = gridValue
        self._validateAmplitudeSize()

    def _validateAmplitudeSize(self):
        assert np.shape(self.gridValue)[1] == len(self.time)
        assert np.shape(self.gridValue)[0] == len(self.frequency)
        assert np.shape(self.gridValue)[2] == len(self.n_range)

    def _initializePlottingBehavior(self):
        self.threshold = 0.5  # Value between 0 and 1.
        self.voxel_size = 20  # Size of a voxel
        self.title = "Voxel Plot"
        self.marker = "s"

    # noinspection PyAttributeOutsideInit
    def setPlottingBehavior(
        self, threshold: float, voxel_size: float, title: str, marker: str = "s"
    ):
        self.threshold = threshold
        self.voxel_size = voxel_size
        self.title = title
        self.marker = marker

    def plot(self):
        output_values = np.abs(self.gridValue)
        maximum, _ = max_in_array(output_values)

        tt, ff, nn = np.meshgrid(self.time, self.frequency, self.n_range, indexing="ij")

        indices = np.argwhere(
            output_values > self.threshold * maximum
        )  # Get indices where condition is met
        values = output_values[
            indices[:, 0], indices[:, 1], indices[:, 2]
        ]  # Extract values at those indices

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Convert indices to integer scalar arrays
        freq_indices = indices[:, 0].astype(int)
        time_indices = indices[:, 1].astype(int)

        ax.scatter(
            self.frequency[freq_indices],
            self.time[time_indices],
            self.n_range[indices[:, 2]],
            c=values,
            marker=self.marker,
            alpha=0.8,
        )
        ax.set_xlabel("n-frequency")
        ax.set_ylabel("time")
        ax.set_zlabel("n-value")

        plt.show()


def create2DPlot(grid: np.array):
    fig, ax = plt.subplots()
    image = ax.matshow(grid, cmap="gray")
    return fig, ax, image


def set2DLabels(ax, grid: np.array, XY_skips: tuple):
    ax.set_xticks(np.arange(0, np.shape(grid)[1], XY_skips[0]))
    ax.set_yticks(np.arange(0, np.shape(grid)[0], XY_skips[1]))


def setTimeFreqAxes(
    ax,
    XY_data: tuple,
    XY_skips: tuple,
    XY_fontsize: tuple,
    xlabel: str = "Time",
    ylabel: str = "Freq",
    title: str = "",
):
    ax.set_xticklabels(XY_data[0][:: XY_skips[0]], fontsize=XY_fontsize[0], rotation=90)
    ax.set_yticklabels(XY_data[1][:: XY_skips[1]], fontsize=XY_fontsize[1])
    labelTimeFreqAxes(XY_fontsize, xlabel, ylabel, title)


def labelTimeFreqAxes(
    XY_fontsize: tuple, xlabel: str = "Time", ylabel: str = "Freq", title: str = ""
):
    plt.xticks(fontsize=XY_fontsize[0])
    plt.yticks(fontsize=XY_fontsize[1])
    plt.xlabel(ylabel)
    plt.ylabel(xlabel)
    plt.title(title)


def generatePixelGridData():
    grid = generateRandomGrid(1000, 1000)
    time = range(1000)
    freq = range(1000)

    return PixelGrid(time, freq, grid)


def generatePixelGridPlotParams(pixelgrid: PixelGrid):
    pixelgrid.setPlottingBehavior((50, 50), (10, 10))


def generatePixelGridExample():
    pixelgrid = generatePixelGridData()
    generatePixelGridPlotParams(pixelgrid)
    return pixelgrid


def generateRandomGrid(X, Y, Z=1):
    grid = np.random.random((X, Y, Z))
    return grid


def max_in_array(data: np.array):
    """
    :param data: 3d/2d array. Finds where the maximum within the array is.
    :return: Returns the maximum value and where in the array the maximum value is at.
    """

    Max_val = np.max(data)  # Finds the maximum within the array.
    Max_Index = np.where(data == Max_val)

    return Max_val, Max_Index


if __name__ == "__main__":
    myPixels = generatePixelGridExample()
    myPixels.plot()

    time = np.arange(0, 1, 1 / 50)
    freq = np.arange(0, 1, 1 / 50)
    n_range = np.arange(1, 5, 1 / 10)
    vox_grid = generateRandomGrid(50, 50, 40)
    vox = VoxelGrid(time, freq, n_range, vox_grid)
    vox.setPlottingBehavior(0.999, 6, "Title")
    vox.plot()
