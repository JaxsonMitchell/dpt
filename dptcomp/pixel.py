import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.animation as animation
import imageio


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
    
    def windowExtract(self, n: int, time: float, frequency: float, sigmaTime: float = 1, sigmaFrequency: float = 1):
        """
        Applies a windowing function to the gridValues attribute of a VoxelGrid object. 

        input:
            n (int) - Exponent for the windowing function
            time (float) - Central time for the window
            frequency (float) - Central frequency for the window
            sigmaTime (float) - Standard deviation for the time window, default is 1
            sigmaFrequency (float) - Standard deviation for the frequency window, default is 1

        output:
            None (None)
        """
        self.gridValue *= np.exp(- (self.time - time) ** (2 * n) / (sigmaTime * 2 * n)) * np.exp(- (self.frequency - frequency) ** (2 * n) / (sigmaFrequency * 2 * n)) 


    def windowSubtract(self, n: int, time: float, frequency: float, sigmaTime: float = 1, sigmaFrequency: float = 1):
        """
        Applies 1 - windowing function to the gridValues attribute of a VoxelGrid object.

        input:
            n (int) - Exponent for the windowing function
            time (float) - Central time for the window
            frequency (float) - Central frequency for the window
            sigmaTime (float) - Standard deviation for the time window, default is 1
            sigmaFrequency (float) - Standard deviation for the frequency window, default is 1

        output:
            None (None)
        """
        self.gridValue *= (1 - np.exp(- abs(self.time - time) ** (2 * n) / (sigmaTime * 2 * n)) * np.exp(- abs(self.frequency - frequency) ** (2 * n) / (sigmaFrequency * 2 * n)))

    
    def whitenSignal(self, PSD: np.ndarray):
        """
        Divides the gridValues attribute of the VoxelGrid object by the square root of the provided n-Power Spectral Density (n-PSD).

        input:
            PSD (np.ndarray) - The Power Spectral Density array

        output:
            None (None)
        """
        PSDGrid = np.tile(np.sqrt(PSD), (len(PSD), 1)).T
        self.gridValue /= PSDGrid


class VoxelGrid:
    CUSTOMCMAP = mcolors.LinearSegmentedColormap.from_list("CCmap", [(1, 1, 1), (0, 0, 0)])

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

        maximum, _ = maxInArray(output_values)

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
            alpha=.8,
            cmap="magma_r",
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
        outputValues = np.abs(self.gridValue)

        flatValues = outputValues.flatten()
        flatIndices = np.indices(outputValues.shape).reshape((3, -1)).T

        topIndices = np.argsort(flatValues)[-N:][::-1]

        topValues = []
        for idx in topIndices:
            freq_idx, time_idx, n_idx = flatIndices[idx]
            topValues.append((self.time[time_idx], self.frequency[freq_idx], self.n_range[n_idx], flatValues[idx]))
        return topValues

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
        
    def plotSlice(
            self,
            index: int,
            time: bool = False,
            nFreq: bool = False,
            chirpOrder: bool = False,
            plot: bool = True,
            fig=None,
            ax=None
        ) -> None:
        """ 2-dimensional representation of chirps. The method takes an index and a kwarg indicating 
        a slice in either time, n-frequency, or chirp order. It is required to set one value to true.
        
        Kwarg arguments
            time
            nFreq
            chirpOrder

        If two values are set to true, it will choose the first value in the order time, nFreq, and chirpOrder
        """
        maximum, _ = maxInArray(self.gridValue)

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        if time:
            im = ax.matshow(np.abs(self.gridValue[index, :, :]), vmax=maximum)
            set2DLabels(ax, self.gridValue[:, index, :], (8, 8))
            setTimeFreqAxes(
                ax,
                (np.round(self.frequency, 4), np.round(self.n_range, 4)),
                (8, 8),
                (12, 12),
                "Chirp Order",
                "n-Frequency",
                f"Chirp Order vs. n-Frequency (time = {round(self.time[index], 4)})",
                12
            )
            ax.set_aspect(len(self.frequency) / (2 * len(self.n_range)))
            plt.tight_layout()
            if plot:
                plt.show()
            return im
        elif nFreq:
            im = ax.matshow(np.abs(self.gridValue[:, index, :]), vmax=maximum)
            set2DLabels(ax, self.gridValue[:, index, :], (8, 8))
            setTimeFreqAxes(
                ax,
                (np.round(self.time, 4), np.round(self.n_range, 4)),
                (8, 8),
                (12, 12),
                "Chirp Order",
                "Time",
                f"Chirp Order vs. Time (n-Frequency = {round(self.frequency[index], 4)})",
                12
            )
            ax.set_aspect(len(self.time) / (2 * len(self.n_range)))
            plt.tight_layout()
            if plot:
                plt.show()
            return im
        elif chirpOrder:
            im = ax.matshow(np.abs(self.gridValue[:, :, index]), vmax=maximum)
            set2DLabels(ax, self.gridValue[:, :, index], (8, 8))
            setTimeFreqAxes(
                ax,
                (np.round(self.time, 4), np.round(self.n_range, 4)),
                (8, 8),
                (12, 12),
                "n-Frequency",
                "Time",
                f"n-Frequency vs. Time (Chirp Order = {round(self.n_range[index], 4)})",
                12
            )
            ax.set_aspect(len(self.time) / (2 * len(self.frequency)))
            plt.tight_layout()
            if plot:
                plt.show()
            return im
        else:
            raise ValueError("Invalid Input. One kwarg from time, nFreq, and chirpOrder must be set to True.")

    def makeVoxelVideo(self, filename: str, interval: int, title: str = "Voxel Grid", XY_skips: tuple = (16, 16)) -> None:
        """
        Method for saving voxel grids as a video where:

        """
        fig, ax = plt.subplots()

        vmin = np.min(self.gridValue)
        vmax = np.max(self.gridValue)

        im = ax.imshow(self.gridValue[:, :, 0], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        fig.colorbar(im, ax=ax, label='Value')

        def update(frame):
            im.set_array(self.gridValue[:, :, frame])
            setTimeFreqAxes(
                ax, XY_data = (self.time, self.frequency), XY_skips = XY_skips,
                XY_fontsize = (14, 14), xlabel = "Time", 
                ylabel = f"{round(self.n_range[frame], 5)}-Frequency",
                title = title
            )
            return [im]

        ani = animation.FuncAnimation(
            fig, update,
            frames=np.shape(self.gridValue)[2],
            interval=interval, blit=True,
        )

        fig.tight_layout()
        ani.save(filename, writer='ffmpeg')


def create2DPlot(grid: np.array):
    fig, ax = plt.subplots()
    image = ax.matshow(grid)
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
    ax.set_xticks(range(0, len(XY_data[0]), XY_skips[0]))
    ax.set_yticks(range(0, len(XY_data[1]), XY_skips[1]))
    
    ax.set_xticklabels(np.round(XY_data[0][::XY_skips[0]], 4), fontsize=XY_fontsize[0], rotation=90)
    ax.set_yticklabels(np.round(XY_data[1][::XY_skips[1]], 4), fontsize=XY_fontsize[1])

    labelTimeFreqAxes(ax, XY_fontsize_labels, xlabel, ylabel, title)


def labelTimeFreqAxes(
    ax, XY_fontsize: int, xlabel: str = "Time", ylabel: str = "Freq", title: str = ""
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


def maxInArray(data: np.array):
    """
    :param data: 3d/2d array. Finds where the maximum within the array is.
    :return: Returns the maximum value and where in the array the maximum value is at.
    """

    maxVal = np.max(data)  # Finds the maximum within the array.
    maxIndex = np.where(data == maxVal)

    return maxVal, maxIndex


def minInArray(data: np.array):
    """
    :param data: 3d/2d array. Finds where the minimim within the array is.
    :return: Returns the minimum value and where in the array the minimum value is at.
    """

    minVal = np.min(data)  # Finds the maximum within the array.
    minIndex = np.where(data == minVal)

    return minVal, minIndex


if __name__ == "__main__":
    myPixels = generatePixelGridExample()
    myPixels.plot()

    time = np.arange(0, 1, 1 / 100)
    freq = np.arange(0, 1, 1 / 100)
    n_range = np.arange(1, 5, 1 / 30)
    vox_grid = generateRandomGrid(len(time), len(freq), len(n_range))
    vox = VoxelGrid(time, freq, n_range, vox_grid)
    vox.makeVoxelVideo("test.mp4", 50, "Test")
    vox.makeVoxelVideo("test.gif", 50, "em Silly")