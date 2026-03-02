"""
@created: 16.03.2021
@author: Fernando Moya Rueda

@copyright: Motion Miners GmbH, Emil-Figge Str. 76, 44227 Dortmund, 2022

@brief: Metrics using GPU
"""

import os
import pickle

# Ignore warnings
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class HARWindows(Dataset):
    """
    classdocs
    """

    def __init__(self, config, csv_file, root_dir, type_dataset="train", transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = config
        self.harwindows = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.type_dataset = type_dataset

    def __len__(self):
        return len(self.harwindows)

    def __getitem__(self, idx):
        """
        get single item

        @param data: index of item in List
        @return window_data: dict with sequence window, label of window, and labels of each sample in window
        """
        window_name = os.path.join(self.root_dir, self.harwindows.iloc[idx, 0])

        f = open(window_name, "rb")
        data = pickle.load(f, encoding="bytes")
        f.close()

        X = data["data"]
        #print("Size input before windows preprocessing:", X.shape)
        X = self.norm_motionminers_real(X, datasource="motionminers_large")

        Y = data["labels"]
        if self.type_dataset == "train":
            X, Y = self._time_warp_speed(X, Y, self.config["sliding_window_length"])
            #X, Y = self._time_warp_speed(X, Y, 1000)
            X = self._time_warp(X)

        identity = data["identity"]
        label_file = data["label_file"]

        window_data = {"data": X, "labels": Y, "identity": identity, "label_file": label_file,
                       "file_name": self.harwindows.iloc[idx, 0], }

        return window_data

    def norm_motionminers_real(self, data, datasource="motionminers_large"):
        # data [L, T, R]

        if datasource == "motionminers_large":
            mean_values = np.array(
                [
                    # Acc
                    -2000.80300,
                    1422.51300,
                    1375.16100,
                    # Gyr
                    6.62400,
                    -7.83100,
                    -6.55000,
                    # Acc
                    226.63100,
                    -3526.73000,
                    -212.65200,
                    # Gyr
                    0.37500,
                    5.60400,
                    -2.00600,
                    # Acc
                    2084.81800,
                    1276.25700,
                    1421.84100,
                    # Gyr
                    2.42100,
                    -2.21200,
                    7.95900,
                ]
            )
            mean_values = np.reshape(mean_values, [1, 1, 18])

            std_values = np.array(
                [
                    # Acc
                    1680.68779,
                    2097.85196,
                    2080.44867,
                    # Gyr
                    772.80695,
                    1232.88517,
                    927.08393,
                    # Acc
                    852.01600,
                    1306.46268,
                    1642.09722,
                    # Gyr
                    380.18336,
                    594.04448,
                    331.90883,
                    # Acc
                    1739.74145,
                    2098.20442,
                    2044.94942,
                    # Gyr
                    827.49874,
                    1270.91930,
                    936.18195,
                ]
            )
            std_values = np.reshape(std_values, [1, 1, 18])

        mean_array = np.repeat(mean_values, data.shape[1], axis=1)
        std_array = np.repeat(std_values, data.shape[1], axis=1)

        max_values = mean_array + 2 * std_array
        min_values = mean_array - 2 * std_array

        data_norm = (data - min_values) / (max_values - min_values)

        data_norm[data_norm > 1] = 1
        data_norm[data_norm < 0] = 0

        # data_norm = (data - mean_array) / std_array

        return data_norm

    def _random_curve(self, window_len: int, sigma=0.05, knot=4):
        """
        Generates a random cubic spline with mean value 1.0.
        This curve can be used for smooth, random distortions of the data, e.g., used for time warping.

        Note: According to T. Um, a cubic splice is not the best approach to generate random curves.
        Other aprroaches, e.g., Gaussian process regression, Bezier curve, etc. are also reasonable.

        :param window_len: Length of the data window (for example, 100 frames), the curve will have this length
        :param sigma: sigma of the curve, the spline deviates from a mean of 1.0 by +- sigma
        :param knot: Number of anchor points
        :return: A 1d cubic spline
        """

        random_generator = np.random.default_rng()

        xx = (np.arange(0, window_len, (window_len - 1) / (knot + 1))).transpose()
        yy = random_generator.normal(loc=1.0, scale=sigma, size=(knot + 2, 1))
        x_range = np.arange(window_len)
        cs_x = CubicSpline(xx, yy)
        return cs_x(x_range).flatten()

    def _time_warp(self, sample: np.ndarray) -> np.ndarray:
        """
        Computes a time warping that using a random cubic curve

        :param sample: sample of shape [1, time, channels]
        :return: augmented sample
        """
        window_len = sample.shape[1]
        num_samples = sample.shape[2]

        time_warp_scale = 0.05

        #
        # Generate new time sampling values using a random curve
        # Generate curve, accumulate timestamps
        #
        timesteps = self._random_curve(window_len, sigma=time_warp_scale)
        tt_cum = np.cumsum(timesteps, axis=0)  # Add intervals to make a cumulative graph
        # Make the last value to have X.shape[0]
        t_scale = (window_len - 1) / tt_cum[-1]
        tt_cum = tt_cum * t_scale

        #
        # Resample
        #
        x_range = np.arange(window_len)
        resampled = np.zeros(sample.shape)
        for s_i in range(num_samples):
            resampled[0, :, s_i] = np.interp(x_range, tt_cum, sample[0, :, s_i].flatten())
            # Clamp first and last value
            resampled[0, 0, s_i] = resampled[0, 0, s_i]
            resampled[0, -1, s_i] = resampled[0, -1, s_i]

        # Return the warped sample
        return resampled

    def _time_warp_speed(self, sample: np.ndarray, annotations: np.ndarray, new_window_len: int):
        """
        Computes a time warping that using a random cubic curve

        :param sample: sample of shape [1, time, channels]
        :return: augmented sample
        """
        # length_rdm = (np.random.randint(low=90, high=sample.shape[1], size=1)) // 2
        #length_rdm = (np.random.randint(low=140, high=150, size=1)) // 2
        length_rdm = (np.random.randint(low=self.config["sliding_window_length"] - 10, high=self.config["sliding_window_length"], size=1)) // 2
        middle = sample.shape[1] // 2
        new_range_rdm = np.arange(middle - length_rdm, middle + length_rdm)
        data = sample[:, new_range_rdm, :]

        window_len = data.shape[1]
        num_samples = data.shape[2]

        #
        # Generate new time sampling values using a random curve
        # Generate curve, accumulate timestamps
        #
        t_sep = (window_len - 1) / (new_window_len - 1)
        range_int = np.arange(0, window_len, t_sep)
        range_int = range_int[:new_window_len]

        #
        # Resample
        #
        x_range = np.arange(window_len)
        resampled = np.zeros((1, new_window_len, data.shape[2]))
        for s_i in range(num_samples):
            resampled[0, :, s_i] = np.interp(range_int, x_range, data[0, :, s_i].flatten())
            # Clamp first and last value
            resampled[0, 0, s_i] = resampled[0, 0, s_i]
            resampled[0, -1, s_i] = resampled[0, -1, s_i]

        middle = sample.shape[1] // 2
        new_range_rdm = np.arange(middle - (new_window_len // 2), middle + (new_window_len // 2))
        new_range_rdm = new_range_rdm[:new_window_len]

        # Return the warped sample
        return resampled, annotations[new_range_rdm, :]
