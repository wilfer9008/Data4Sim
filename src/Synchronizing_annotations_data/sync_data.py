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

    # Get the annotations fropm the csv file
    annotation = get_csv(path_to_annotation)

    # Slice the annotations accoridng to the valid range given in the sync_info
    annotation = annotation[recording_info["annotation_sync_ranges"][0][0]:recording_info["annotation_sync_ranges"][0][1], :]

    # Resampling to the IMU sampling
    rate_time = (annotation.shape[0] / 30) / (((length_sensors * 10) / 1000) / 30)
    annotation_time = np.arange(0, annotation.shape[0], rate_time / 100)

    if len(annotation_time) == length_sensors:
        annotation = annotation[annotation_time.astype(int)]
    else:
        annotation = annotation[annotation_time[:-1].astype(int)]

    return annotation

def sync_data(subject_id: int, imu_set_id: int, path_folder: str, path_to_annotation: str):
    """
    sync the annotations with the data sampling time

    @param subject_id: subject ID
    @param imu_set_id: Set ID
    @param path_folder: path of the root folder with all of the data
    @param path_to_annotation: path of annotation file (as there are many annotations for Data4Sim)

    @return: 
    """

    # get the metadata of the subjects
    recording_info = MEASUREMENTS_INFO[subject_id][imu_set_id]

    # get recordings and data
    measurement_imu_data = get_csv(path_folder + "0_raw_data/" + recording_info["imu_file_name"])
    measurement_imu_data_interval = np.loadtxt(path_folder + "0_raw_data/" + recording_info["imu_time_interval_raw"], delimiter=",", skiprows=0)
    measurement_ble_data = get_csv(path_folder + "0_raw_data/" + recording_info["ble_file_name"])
    measurement_dyn_ble_usage_arr = get_csv(path_folder + "0_raw_dyn_beacon_data/usage_arr/" + recording_info["dyn_usage_arr"])
    measurement_dyn_ble_closeness_arr = get_csv(path_folder + "0_raw_dyn_beacon_data/closeness_arr/" + recording_info["dyn_closeness_arr"])
    measurement_ble_data = get_csv(path_folder + "0_raw_data/" + recording_info["ble_file_name"])
    measurement_ble_headers = get_headers(path_folder + "0_raw_data/" + recording_info["ble_file_name"])
    measurement_activity_MM_predictions = get_csv(
        path_folder + "1_MM_predictions_mpi/" + recording_info["predictionsMM_file_name"]
    )
    measurement_activity_MM_CAR_predictions = get_csv(
        path_folder + "1_MM_predictions_car/" + recording_info["predictionsMM_file_name"]
    )
    measurement_activity_MM_heights_predictions = get_csv(
        path_folder + "1_MM_predictions_mpi_heights/" + recording_info["predictionsMM_file_name"]
    )

    # get the predictions of the fingerprinting
    data_fingerprinting_reg_uuid = []
    data_fingerprinting_start_time = []
    data_fingerprinting_end_time = []
    with open(path_folder + "0_raw_data_fingerprint_mpi/" + recording_info["predictionsMM_fingerprinting_file_name"], newline='') as f:
        reader = csv.reader(f)
        data_fingerprinting = list(reader)

    # getting the range per location prediction
    for df in range(1, len(data_fingerprinting)):
        data_fg = data_fingerprinting[df][0].split(';')
        if (float(data_fg[1]) * 1000) > measurement_imu_data_interval[0] and (float(data_fg[2]) * 1000) < measurement_imu_data_interval[1]:
            data_fingerprinting_reg_uuid.append(data_fg[0])
            data_fingerprinting_start_time.append(float(data_fg[1]) * 1000)
            data_fingerprinting_end_time.append(float(data_fg[2]) * 1000)

    # mapp the range per class prediction to 100Hz sampling
    data_fingerprinting_start_time = np.array(data_fingerprinting_start_time)
    data_fingerprinting_end_time = np.array(data_fingerprinting_end_time)
    fingerprinting_idxs = np.arange(data_fingerprinting_start_time[0], data_fingerprinting_end_time[-1], 100)
    labels_fingerprinting = []
    for df in range(fingerprinting_idxs.shape[0]):
        idx_fingerprint = (fingerprinting_idxs[df] >= data_fingerprinting_start_time) * (fingerprinting_idxs[df] < data_fingerprinting_end_time)
        idx_fingerprint = np.argmax(idx_fingerprint)
        #labels_fingerprinting.append("{}".format(int(fingerprinting_idxs[df])) + ';' +data_fingerprinting[idx_fingerprint + 1][0].split(';')[0])
        labels_fingerprinting.append(["{}".format(int(fingerprinting_idxs[df])), data_fingerprinting[idx_fingerprint + 1][0].split(';')[0]])

    # change the range according to the sync movements
    measurement_imu_data = measurement_imu_data[
        recording_info["imu_data_sync_ranges"][0][0] : recording_info["imu_data_sync_ranges"][0][1], :
    ].astype(int)
    measurement_activity_MM_predictions = measurement_activity_MM_predictions[
        recording_info["imu_data_sync_ranges"][0][0] : recording_info["imu_data_sync_ranges"][0][1], :
    ].astype(int)
    measurement_activity_MM_CAR_predictions = measurement_activity_MM_CAR_predictions[
        recording_info["imu_data_sync_ranges"][0][0] : recording_info["imu_data_sync_ranges"][0][1], :
    ].astype(int)
    measurement_activity_MM_heights_predictions = measurement_activity_MM_heights_predictions[
        recording_info["imu_data_sync_ranges"][0][0] : recording_info["imu_data_sync_ranges"][0][1], :
    ].astype(int)
    measurement_ble_data = measurement_ble_data[
        recording_info["imu_data_sync_ranges"][0][0] // 10 : recording_info["imu_data_sync_ranges"][0][1] // 10, :
    ].astype(np.float16)
    measurement_dyn_ble_usage_arr = measurement_dyn_ble_usage_arr[
        recording_info["imu_data_sync_ranges"][0][0] // 10 : recording_info["imu_data_sync_ranges"][0][1] // 10, :
    ].astype(np.float16)
    measurement_dyn_ble_closeness_arr = measurement_dyn_ble_closeness_arr[
        recording_info["imu_data_sync_ranges"][0][0] // 10 : recording_info["imu_data_sync_ranges"][0][1] // 10, :
    ].astype(np.float16)

    # get annotations
    annotations = sync_annotations(
        path_to_annotation, recording_info, length_sensors=measurement_imu_data.shape[0]
    ).astype(int)
    print("annotations", annotations.shape)


    # Saving IMU data
    filename = path_folder + "2_data/" + recording_info["imu_file_name"]
    with open(filename, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(headers)
        for d in measurement_imu_data:
            spamwriter.writerow(d)

    # Saving BLE Data
    filename = path_folder + "2_data/" + recording_info["ble_file_name"]
    with open(filename, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(measurement_ble_headers)
        for d in measurement_ble_data:
            spamwriter.writerow(d)

    # Saving the annotations with 100Hz
    # filename = path_folder + "3_sync_labels/" + recording_info["labels_file_name"]
    # with open(filename, "w", newline="") as csvfile:
    #    spamwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    #    spamwriter.writerow(headers_annotations)
    #    for d in annotations:
    #        spamwriter.writerow(d)

    # Saving MM predictions
    header_class = [
        "Null",
        "Walking",
        "Standing",
        "Handling",
        "Driving",
    ]

    filename = path_folder + "2_data/" + recording_info["predictionsMM_file_name"]
    with open(filename, "w", newline="") as csvfile:
       spamwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
       spamwriter.writerow(header_class)
       for d in measurement_activity_MM_predictions:
           spamwriter.writerow(d)


    # Saving the predictions from CAR model MM
    # First, getting the label names
    with open(path_to_annotation, 'r') as f:
        header_class = f.readline()[:-2].split(",")

    filename = path_folder + "2_data/" + recording_info["predictionsMM_car_file_name"]
    with open(filename, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(header_class)
        for d in measurement_activity_MM_CAR_predictions:
            spamwriter.writerow(d)

    # Saving the MM heights predictions
    header_class = [
        "Null",
        "HANDLE_UPWARDS",
        "HANDLE_CENTRED",
        "HANDLE_DOWNWARDS",
    ]

    filename = path_folder + "2_data/" + recording_info["predictionsMM_heights_file_name"]
    with open(filename, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(header_class)
        for d in measurement_activity_MM_heights_predictions:
            spamwriter.writerow(d)

    # Saving the usage dynamic beacons located on the CARTs
    header_class = [
        "55",
        "56",
        "57"
    ]

    filename = path_folder + "2_data/" + recording_info["dyn_usage_arr"]
    with open(filename, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(header_class)
        for d in measurement_dyn_ble_usage_arr:
            spamwriter.writerow(d)

    # Saving the close dynamic beacons located on the CARTs
    header_class = [
        "55",
        "56",
        "57"
    ]

    filename = path_folder + "2_data/" + recording_info["dyn_closeness_arr"]
    with open(filename, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(header_class)
        for d in measurement_dyn_ble_closeness_arr:
            spamwriter.writerow(d)

    # Saving finger printing predictions
    header_class = [
        "Time",
        "Region"
    ]
    filename = path_folder + "1_MM_predictions_mpi_fingerprinting/" + recording_info["predictionsMM_fingerprinting_sync_file_name"]
    with open(filename, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(header_class)
        for d in labels_fingerprinting:
            spamwriter.writerow(d)
            
    return


def sync_only_labels(subject_id: int, imu_set_id: int, path_to_annotation: str):
    """
    sync the annotations with the data sampling time of the IMU and of the BLE data

    @param subject_id: Subject ID
    @param imu_set_id: Set ID
    @param path_to_annotation: path of annotaiton file (as there are many annotations for Data4Sim)

    @return: 
    """

    # get range of valid data for IMU and annotations stored in sync_info
    recording_info = MEASUREMENTS_INFO[subject_id][imu_set_id]

    # get recordings and data
    measurement_imu_data = get_csv("0_raw_data/" + recording_info["imu_file_name"])

    # change the range according to the sync movements
    measurement_imu_data = measurement_imu_data[
        recording_info["imu_data_sync_ranges"][0][0] : recording_info["imu_data_sync_ranges"][0][1], :
    ].astype(int)

    # get annotations
    annotations = sync_annotations(path_to_annotation, recording_info, length_sensors=measurement_imu_data.shape[0]).astype(int)
    print("annotations 100Hz", annotations.shape)


    # get the headers of the annotations
    with open(path_to_annotation, 'r') as f:
        header_class = f.readline()[:-1].split(",")

    # store the annotations with 100Hz in sync with the IMU data
    filename = "3_sync_labels/100Hz_" + recording_info["labels_file_name"]
    with open(filename, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(header_class)
        for d in annotations:
            spamwriter.writerow(d)

    # store the annotations with 10Hz in sync with the IMU data
    annotations_10 = annotations[np.arange(0, annotations.shape[0], 10)].astype(int)
    print("annotations 10Hz", annotations_10.shape)

    filename = "3_sync_labels/10Hz_" + recording_info["labels_file_name"]
    with open(filename, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(header_class)
        for d in annotations_10:
            spamwriter.writerow(d)

    return

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Plot IMU raw data")
    #parser.add_argument("subject_id", type=int, help="ID of the subject")
    #parser.add_argument("imu_set_id", type=int, help="ID of the IMU set")
    #parser.add_argument("path_to_annotation", type=str, help="path to store the csv")
    #args = parser.parse_args()

    # Set the path where all the files are located
    # Here as example, the folder, I use
    path_folder = "/home/fernando/Documents/Repositories/checkouts/data4sim/sync_data/"
    label_target = "Revised_Annotation__CC04_Sub-Activity - Left Hand" #example

    # Run for sync all the files of a given subject and set id
    #sync_data(args.subject_id, args.imu_set_id, args.path_to_annotation)

    # Run for sync only the annotation file of a given subject and set id
    for ss in range(1, 19):
        for ssr in range(1, 3):
            # For Sync the data
            #sync_data(
            #    ss,
            #    ssr,
            #    path_folder,
            #    path_folder + "0_annotations/Main_Activity/S{:02d}_Activity.csv".format(
            #        ss
            #    ),
            #)
            sync_only_labels(
                ss, 
                ssr, 
                path_folder + "0_annotations/Video__Revised_Annotations/{}/{}__S{:02d}.csv".format(
                    label_target,
                    label_target,
                    ss
                ),
                )

    # Run for sync only the annotation file of a given subject and set id
    #sync_only_labels(args.subject_id, args.imu_set_id, args.path_to_annotation)

    # example
    # python3.11 sync_data.py 1 2 "path-to-annotations/S01_Activity.csv"
