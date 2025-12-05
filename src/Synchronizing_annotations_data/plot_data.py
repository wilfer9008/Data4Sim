"""Plot IMU data for a specific measurement.

Created:
    21.08.2025

Author:
    Fernando Moya Rueda

Copyright:
    MotionMiners GmbH, Emil-Figge Str. 80, 44227 Dortmund, 2024
"""

import argparse
import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np

from sync_info import MEASUREMENTS_INFO

last_click_val = ""

headers = [
    "Time",
    "AccX_L",
    "AccY_L",
    "AccZ_L",
    "GyrX_L",
    "GyrY_L",
    "GyrZ_L",
    "AccX_T",
    "AccY_T",
    "AccZ_T",
    "GyrX_T",
    "GyrY_T",
    "GyrZ_T",
    "AccX_R",
    "AccY_R",
    "AccZ_R",
    "GyrX_R",
    "GyrY_R",
    "GyrZ_R",
]

headers_annotations = [
    "Synchronization",
    "Tick off / confirm",
    "Scan",
    "Pull",
    "Push",
    "Handling upwards",
    "Handling centred",
    "Handling downwards",
    "Walking",
    "Standing",
    "Sitting",
    "ANOTHER ACTIVITY",
    "ACTIVITY UNKNOWN",
]


def get_csv(path, skiprows=1):
    """
    gets data from a csv file from a given path.

    returns a numpy array

    @param path: path to file
    @param skiprows: skiprows in case annotations contain a header

    @return: numpy array
    """

    data = np.loadtxt(path, delimiter=",", skiprows=skiprows)
    return data


def get_headers(path):
    """
    gets headers of the BLE data as these headers might change depending on the recording
    returns a numpy array

    @param path: path to file
    @param skiprows: skiprows in case annotations contain a header

    @return: list of strs with headers
    """

    with open(path, 'r') as dest_f:
        data_iter = csv.reader(dest_f,
                               delimiter=",",
                               quotechar='"')
        data = [data for data in data_iter]
    return data[0]

def sync_annotations(path_to_annotation, recording_info, length_sensors):
    """
    sync the annotations with the data sampling time

    @param path_to_annotation: path to file
    @param recording_info: recording info
    @param length_sensors: length of IMU recoridngs

    @return: np array with annotations sync with data shape
    """

    annotation = get_csv(path_to_annotation)

    annotation = annotation[recording_info["annotation_sync_ranges"][0][0]:recording_info["annotation_sync_ranges"][0][1], :]

    rate_time = (annotation.shape[0] / 30) / (((length_sensors * 10) / 1000) / 30)
    annotation_time = np.arange(0, annotation.shape[0], rate_time / 100)

    if len(annotation_time) == length_sensors:
        annotation = annotation[annotation_time.astype(int)]
    else:
        annotation = annotation[annotation_time[:-1].astype(int)]

    return annotation

def plot_data(subject_id: int, imu_set_id: int, path_to_annotation: str, range_plot = (10000, 100000)):
    """
    sync the annotations with the data sampling time

    @param subject_id: subject ID
    @param imu_set_id: Set ID
    @param path_to_annotation: path of annotaiton file (as there are many annotations for Data4Sim)

    @return: 
    """

    recording_info = MEASUREMENTS_INFO[subject_id][imu_set_id]

    # get recordings and data
    measurement_imu_data = get_csv("0_raw_data/" + recording_info["imu_file_name"])
    measurement_ble_data = get_csv("0_raw_data/" + recording_info["ble_file_name"])
    measurement_ble_headers = get_headers("0_raw_data/" + recording_info["ble_file_name"])

    # change the range according to the sync movements
    measurement_imu_data = measurement_imu_data[recording_info["imu_data_sync_ranges"][0][0]:recording_info["imu_data_sync_ranges"][0][1], :].astype(int)
    measurement_ble_data = measurement_ble_data[
                           recording_info["imu_data_sync_ranges"][0][0] // 10:
                           recording_info["imu_data_sync_ranges"][0][1] // 10, :].astype(np.float16)

    # get annotations
    annotations = sync_annotations(path_to_annotation, recording_info, length_sensors=measurement_imu_data.shape[0]).astype(int)
    print("annotations", annotations.shape)

    time_ble_100 = np.arange(0, measurement_ble_data.shape[0], measurement_ble_data.shape[0] / measurement_imu_data.shape[0]).astype(int)
    measurement_ble_data_100 = measurement_ble_data[time_ble_100]

    # range for plotting
    measurement_ble_data_100 = measurement_ble_data_100[range_plot[0]: range_plot[1], :]
    measurement_imu_data = measurement_imu_data[range_plot[0]: range_plot[1], :]

    # Plotting
    data_range_init = 0
    data_range_end = measurement_imu_data.shape[0]
    time_x = np.arange(data_range_init, data_range_end)

    activity_class = np.argmax(annotations, axis=1)
    print("activity_class", activity_class.shape)

    T = np.linalg.norm(measurement_imu_data[data_range_init:data_range_end, [7, 8, 9]], axis=1)  # / 4000
    L = np.linalg.norm(measurement_imu_data[data_range_init:data_range_end, [1, 2, 3]], axis=1)  # / 4000
    R = np.linalg.norm(measurement_imu_data[data_range_init:data_range_end, [13, 14, 15]], axis=1)  # / 4000

    # Plotting
    fig = plt.figure()
    axis_list = []
    plot_list = []
    axis_list.append(fig.add_subplot(211))
    axis_list.append(fig.add_subplot(212))

    plot_list.append(axis_list[0].plot(time_x, L + 0000, "-r", label="L", linewidth=1.5))
    plot_list.append(axis_list[0].plot(time_x, T + 10000, "-g", label="T", linewidth=0.5))
    plot_list.append(axis_list[0].plot(time_x, R + 20000, "-b", label="L", linewidth=0.5))

    #plot_list.append(axis_list[0].plot(time_x, activity_class * 1000, "-g",
    # label="T", linewidth=1.5))
    plot_list.append(axis_list[1].plot(time_x, measurement_ble_data_100[:, 0], "-m", label=measurement_ble_headers[0], linewidth=1.5))
    plot_list.append(axis_list[1].plot(time_x, measurement_ble_data_100[:, 57] + 20, "-c", label=measurement_ble_headers[57], linewidth=1.5))
    plot_list.append(axis_list[1].plot(time_x, measurement_ble_data_100[:, 114] + 40, "-g", label=measurement_ble_headers[114], linewidth=1.5))
    plot_list.append(axis_list[1].plot(time_x, measurement_ble_data_100[:, 1] + 100, "-m", label=measurement_ble_headers[1], linewidth=1.5))
    plot_list.append(axis_list[1].plot(time_x, measurement_ble_data_100[:, 58] + 120, "-c", label=measurement_ble_headers[58], linewidth=1.5))
    plot_list.append(axis_list[1].plot(time_x, measurement_ble_data_100[:, 115] + 140, "-g", label=measurement_ble_headers[115], linewidth=1.5))
    #plot_list.append(axis_list[0].plot(time_x, L - 2000, "-c", label="L", linewidth=1.5))

    # axis_list[0].set_ylim((0, 4))
    axis_list[0].set_xlim((0, data_range_end))
    axis_list[1].set_xlim((0, data_range_end))
    #axis_list[0].legend(loc="upper left", fontsize=10, shadow=True, fancybox=True, borderpad=1)
    #axis_list[1].legend(loc="upper left", fontsize=10, shadow=True, fancybox=True, borderpad=1)

    #fig.suptitle(f"{subject_id=} : {imu_set_id=}", fontsize=12)
    #fig.tight_layout()
    #fig.canvas.draw()

    #plt.grid()
    plt.show()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot IMU raw data")
    parser.add_argument("subject_id", type=int, help="ID of the subject")
    parser.add_argument("imu_set_id", type=int, help="ID of the IMU set")
    parser.add_argument("path_to_annotation", type=str, help="path to store the csv")
    args = parser.parse_args()

    # Run for sync only the annotation file of a given subject and set id
    # it can be also be given the range to plot the data , e.g., (10000, 100000)
    plot_data(args.subject_id, args.imu_set_id, args.path_to_annotation, range_plot = (0, -1))

    # example
    # python3.11 plot_data.py 1 2 "path-to-annotations/S01_Activity.csv"