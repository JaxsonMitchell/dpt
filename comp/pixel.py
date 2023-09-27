import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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
        self.__XY_font_labels = 16

    # noinspection PyAttributeOutsideInit
    def setPlottingBehavior(
        self, XY_skips: tuple, XY_fontsize: tuple, decimals: int = 8, XY_font_labels: int = 16
    ):
        self.__XY_skips = XY_skips
        self.__XY_fontsize = XY_fontsize
        self.decimals = decimals
        self.__XY_font_labels = XY_font_labels

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
            self.__XY_font_labels
        )
        ax.set_aspect(len(self.time) / (2 * len(self.frequency)))
        plt.show()


class VoxelGrid:
    CUSTOMCMAP = mcolors.LinearSegmentedColormap.from_list("CCmap", [(0.5, 0.5, 0.5), (0, 0, 0)])

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
        self.region_of_interest = None

    # noinspection PyAttributeOutsideInit
    def setPlottingBehavior(
        self, threshold: float, voxel_size: float, title: str, marker: str = "s"
    ):
        self.threshold = threshold
        self.voxel_size = voxel_size
        self.title = title
        self.marker = marker

    def setRegionOfInterest(self, frequency_range, time_range, n_range):
        """
        Set the region of interest for plotting.
        Parameters:
            - frequency_range: A tuple (min_freq, max_freq) specifying the frequency range.
            - time_range: A tuple (min_time, max_time) specifying the time range.
            - n_range: A tuple (min_n, max_n) specifying the n range.
        """
        self.region_of_interest = {
            "frequency": frequency_range,
            "time": time_range,
            "n": n_range
        }

    def plot(self):
        output_values = np.abs(self.gridValue)

        # Apply region of interest filtering if specified
        if self.region_of_interest is not None:
            freq_range = self.region_of_interest["frequency"]
            time_range = self.region_of_interest["time"]
            n_range = self.region_of_interest["n"]

            freq_mask = (self.frequency >= freq_range[0]) & (self.frequency <= freq_range[1])
            time_mask = (self.time >= time_range[0]) & (self.time <= time_range[1])
            n_mask = (self.n_range >= n_range[0]) & (self.n_range <= n_range[1])

            output_values[~(freq_mask[:, None, None] & time_mask[None, :, None] & n_mask[None, None, :])] = 0

        maximum, _ = max_in_array(output_values)

        indices = np.argwhere(output_values > self.threshold * maximum)
        values = output_values[indices[:, 0], indices[:, 1], indices[:, 2]]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        left, bottom, width, height = 0.0005, 0.025, .99, .95
        ax.set_position([left, bottom, width, height])

        freq_indices = indices[:, 0].astype(int)
        time_indices = indices[:, 1].astype(int)

        ax.scatter(
            self.frequency[freq_indices],
            self.time[time_indices],
            self.n_range[indices[:, 2]],
            c=values,
            marker=self.marker,
            alpha=0.8,
            cmap=self.CUSTOMCMAP,
        )
        ax.set_xlabel("n-frequency", fontsize=16)
        ax.set_ylabel("time", fontsize=16)
        ax.set_zlabel("n-value", fontsize=16)
        ax.set_title(self.title, fontsize=16)

        plt.show()

    def maxValue(self):
        """ Finds the maximum value of a Voxel grid. """
        maxIndex = np.where(self.gridValue == np.max(self.gridValue))
        indicesList = [
            (t, f, n) for t, f, n in zip(maxIndex[0], maxIndex[1], maxIndex[2])
        ]
        return indicesList


    def maxValues(self, N: int = 1):
        """ Finds the top N values and returns [(time, n-freq, n-value, amplitude) for topN values] """
        output_values = np.abs(self.gridValue)

        flat_values = output_values.flatten()
        flat_indices = np.indices(output_values.shape).reshape((3, -1)).T

        top_indices = np.argsort(flat_values)[-N:][::-1]

        top_values = []
        for idx in top_indices:
            freq_idx, time_idx, n_idx = flat_indices[idx]
            top_values.append((self.time[time_idx], self.frequency[freq_idx], self.n_range[n_idx], flat_values[idx]))
        return top_values

    def maxValuesByBatch(self, N: int = 1, batch_size: int = 100):
        """ Find the maximum N values by processing the voxel grid in smaller batches.
         
        Working with huge datasets such as the datasets obtained from analyzing LIGO data would
        have millions of voxels, processing them all at once ends up running into memory issues. 
        
        The indices get flattened in both examples to make it simpler to sort everything. """
        output_values = np.abs(self.gridValue)
        flat_values = output_values.flatten()
        flat_indices = np.indices(output_values.shape).reshape((3, -1)).T

        num_batches = (len(flat_indices) + batch_size - 1) // batch_size

        top_values = []

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = (batch_num + 1) * batch_size
            batch_indices = flat_indices[start_idx:end_idx]
            batch_values = flat_values[start_idx:end_idx]

            batch_sorted_indices = np.argsort(batch_values)[-N:][::-1]

            for idx in batch_sorted_indices:
                freq_idx, time_idx, n_idx = batch_indices[idx]
                top_values.append((self.time[time_idx], self.frequency[freq_idx], self.n_range[n_idx], batch_values[idx]))

        top_N_values = sorted(top_values, key=lambda x: x[3], reverse=True)[0:N]

        return top_N_values

def create2DPlot(grid: np.array):
    fig, ax = plt.subplots()
    image = ax.matshow(grid, cmap="gray_r")
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
    XY_fontsize_labels: int = 14
):
    ax.set_xticklabels(XY_data[0][:: XY_skips[0]], fontsize=XY_fontsize[0], rotation=90)
    ax.set_yticklabels(XY_data[1][:: XY_skips[1]], fontsize=XY_fontsize[1])
    labelTimeFreqAxes(ax, XY_fontsize_labels, xlabel, ylabel, title)


def labelTimeFreqAxes(
    ax, XY_fontsize: tuple, xlabel: str = "Time", ylabel: str = "Freq", title: str = ""
):
    ax.set_xlabel(xlabel, fontsize=XY_fontsize)
    ax.set_ylabel(ylabel, fontsize=XY_fontsize)
    ax.set_title(title, fontsize=18)


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
    vox.setPlottingBehavior(0.9, 6, "Title")
    # vox.setRegionOfInterest((0, 1), (0, .333), (1, 2))
    vox.plot()

    def custom_function(x, y, z):
        return x + y + z

    x_range = np.linspace(0, 10, 20) 
    y_range = np.linspace(0, 10, 20)
    z_range = np.linspace(0, 10, 20)

    
    x, y, z = np.meshgrid(x_range, y_range, z_range)

    grid_values = custom_function(x, y, z)

    voxel_grid = VoxelGrid(x_range, y_range, z_range, grid_values)

    voxel_grid.setPlottingBehavior(threshold=0.6, voxel_size=5, title="Voxel Plot")

    voxel_grid.plot()
