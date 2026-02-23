"""
@created: 07.05.2024
@author: Fernando Moya Rueda

@copyright: Motion Miners GmbH, Emil-Figge Str. 76, 44227 Dortmund, 2022

@brief: For listing and preparing the CAR dataset
"""

import json

import json
import os
import numpy as np
#from projects_preprocessing import Preprocessing
from annotations import Annotation
import matplotlib.pyplot as plt
import csv

# from motionminers.imu.imu_sequence_containers.exls3_bin_imu_sequence_container import (EXLS3BinIMUSequenceContainer)

headers = [
    "Time",
    "AccX_L",
    "AccY_L",
    "AccZ_L",
    "GyrX_L",
    "GyrY_L",
    "GyrZ_L",
    "MagX_L",
    "MagY_L",
    "MagZ_L",
    "AccX_T",
    "AccY_T",
    "AccZ_T",
    "GyrX_T",
    "GyrY_T",
    "GyrZ_T",
    "MagX_T",
    "MagY_T",
    "MagZ_T",
    "AccX_R",
    "AccY_R",
    "AccZ_R",
    "GyrX_R",
    "GyrY_R",
    "GyrZ_R",
    "MagX_R",
    "MagY_R",
    "MagZ_R",
]

class Project(object):
    '''
    classdocs
    '''

    def __init__(self, root, path):
        '''
        Constructor
        '''
        self.root = root
        self.project_path = path
        if len(list(self.project_path.split("/"))) == 3:
            self.name = self.project_path.split("/")[1]
        elif len(list(self.project_path.split("/"))) == 4:
            self.name = self.project_path.split("/")[2]
        elif len(list(self.project_path.split("/"))) == 6:
            self.name = self.project_path.split("/")[4]
        self.annotations_v0 = None
        self.raw_annotations_v0 = None
        self.raw_annotations_v1 = None
        self.raw_data = None
        self.annotations_v1 = Annotation()
        self.annotations_v1.set_scheme_path("../scheme/scheme_V2.json")

        print("Getting the data from ", path)
        print("Getting the labels from ", self.root + self.project_path + "/RawData/Sync_V1/" + self.name + "_labels.csv")

        if not os.path.exists(self.root + self.project_path + "/RawData/Sync_V1/" + self.name + "_labels.csv"):
            print("Not existing labels")
            self.get_annotations_v0()
            self.get_annotations_v1()
            self.get_recording_mpi()
            self.annotations_v1.crop_annotations()
            self.annotate_data()
            self.save_sync_data()
        else:
            print("Existing labels")
            annotations = self.get_csv(
                self.root + self.project_path + "/RawData/Sync_V1/" + self.name + "_labels.csv")
            self.annotations_v1.set_sync_annotations(annotations)
            self.raw_data = self.get_csv(
                self.root + self.project_path + "/RawData/Sync_V1/" + self.name + "_data.csv")

        return


    def get_project_info(self):

        return

    def set_config(self):

        config = {'name': self.name,
                  'imu_original': "/RawData/Sensors.zip",
                  'annotated_imu_data': "/RawData/Sync_V1/" + self.name + "_data.csv",
                  'annotated_labels': "/RawData/Sync_V1/" + self.name + "_labels.csv"}

        return

    def load_annotated_sync_data(self):
        annotations = self.get_csv(
            self.root + self.project_path + "/RawData/Sync_V1/" + self.name + "_labels.csv")
        self.annotations_v1.set_sync_annotations(annotations)
        self.raw_data = self.get_csv(
            self.root + self.project_path + "/RawData/Sync_V1/" + self.name + "_data.csv")
        return

    def get_annotations_v0(self):
        old_classes = {0: "NULL", 1: "IGNORE", 2: "WALK", 3: "STAND", 4: "HANDLE", 5: "DRIVE", 6: "SIT"}
        if os.path.exists(self.root + self.project_path + "/RawData/Sync_V0/" + self.name + "_labels.csv"):
            self.annotations_v0 = self.get_csv(self.root + self.project_path + "/RawData/Sync_V0/" + self.name + "_labels.csv")
        elif os.path.exists(self.root + self.project_path + "/RawData/Sync_V0/" + self.name + "_labels_diverse.csv"):
            self.annotations_v0 = self.get_csv(
                self.root + self.project_path + "/RawData/Sync_V0/" + self.name + "_labels_diverse.csv")
        if os.path.exists(self.root + self.project_path + "/RawData/Sync_V0/" + self.name + "_activitylabels_extracted.csv"):
            self.raw_annotations_v0 = self.get_csv(self.root + self.project_path + "/RawData/Sync_V0/" + self.name + "_activitylabels_extracted.csv", skiprows=0)

        return

    def get_annotations_v1(self):

        self.raw_annotations_v1 = self.get_csv(self.root + self.project_path + "/Annotations_V1/" + self.name + "_labels.csv")
        self.annotations_v1.set_annotations(data=self.raw_annotations_v1, path=self.root + self.project_path)

        return self.raw_annotations_v1

    def get_csv(self, path, skiprows=1):
        """
        gets data from a csv file from a given path.

        returns a numpy array

        @param path: path to file
        @param skiprows: skiprows in case annotations contain a header

        @return: numpy array
        """

        data = np.loadtxt(path, delimiter=",", skiprows=skiprows)
        return data

    def get_recording_mpi(self):

        if not os.path.exists(self.root + self.project_path + "/RawData/Sensors.zip"):
            import sys
            sys.path.insert(0, '/media/fernando/femo/Documents/Local_Projects/ErgoCom/miners-core-lib/MotionMinersSensors/src/motionminers/sensors/imus/')
            from imu_data import IMUSequenceContainer
            imu_sequence_container = IMUSequenceContainer()
            ser_file = imu_sequence_container.read_exls3(path=self.root + self.project_path + "RawData/Sensors/")
            self.raw_data = imu_sequence_container.get_data()
        else:
            imu_sequence_container = EXLS3BinIMUSequenceContainer(
                sensor_positions={"L", "T", "R"}, sensor_types={"ACC", "GYR"}
            )

            imu_sequence_container.read_exls(file_handle=self.root + self.project_path + "/RawData/Sensors.zip")
            self.raw_data = imu_sequence_container.get_data(sensor_positions={"L", "T", "R"}, sensor_types={"ACC", "GYR"})

        return self.raw_data

    def annotate_data(self):

        self.annotations_v1.sync_annotations(length_sensors=self.raw_data.shape[0])

        return

    def save_sync_data(self):

        filename = self.root + self.project_path + "/RawData/Sync_V1/" + self.name + "_data.csv"
        with open(filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(headers)
            for d in self.raw_data:
                spamwriter.writerow(d)

        self.annotations_v1.save_sync_annotations(self.root + self.project_path + "/RawData/Sync_V1/" + self.name + "_labels.csv")

        return

    def mouse_event(self, event):
        print("x: {} and y: {}".format(int(event.xdata), int(event.ydata)))

    def visualize(self):
        fig = plt.figure()
        cid = fig.canvas.mpl_connect("button_press_event", self.mouse_event)
        axis_list = []
        plot_list = []
        axis_list.append(fig.add_subplot(311))
        axis_list.append(fig.add_subplot(312))
        axis_list.append(fig.add_subplot(313))

        plot_list.append(axis_list[0].plot([], [], "-r", label="T", linewidth=0.15)[0])
        plot_list.append(axis_list[0].plot([], [], "-b", label="L", linewidth=0.20)[0])
        plot_list.append(axis_list[0].plot([], [], "-g", label="R", linewidth=0.20)[0])

        plot_list.append(axis_list[1].plot([], [], "-r", label="T", linewidth=0.15)[0])
        plot_list.append(axis_list[1].plot([], [], "-b", label="LR", linewidth=0.30)[0])
        plot_list.append(axis_list[1].plot([], [], "-c", label="P", linewidth=0.40)[0])
        plot_list.append(axis_list[1].plot([], [], "-g", label="P", linewidth=0.50)[0])

        plot_list.append(axis_list[2].plot([], [], "-r", label="P", linewidth=0.15)[0])
        plot_list.append(axis_list[2].plot([], [], "-b", label="A", linewidth=0.30)[0])

        #  AccX,AccY,AccZ, GyrX,GyrY,GyrZ, MagX,MagY,MagZ
        # data [T, 28] with L [:, 1:7] T [:, 7:13] R [:, 13:]
        # 1,    2,      3,    4,    5,    6,
        # AccX, AccY, AccZ, GyrX, GyrY, GyrZ,
        # 7,     8,    9,     10,    11,   12,
        # AccX, AccY, AccZ, GyrX, GyrY, GyrZ,
        # 13,    14,   15,    16,   17,  18,
        # AccX, AccY, AccZ, GyrX, GyrY, GyrZ,
        data_range_init = 0
        data_range_end = self.raw_data.shape[0]
        time_x = np.arange(data_range_init, data_range_end)

        print("Range init {} end {}".format(data_range_init, data_range_end))
        T = np.linalg.norm(self.raw_data[data_range_init:data_range_end, 7:13], axis=1)
        L = np.linalg.norm(self.raw_data[data_range_init:data_range_end, [1, 2, 3, 4, 5, 6]], axis=1)
        R = np.linalg.norm(self.raw_data[data_range_init:data_range_end, [13, 14, 15, 16, 17, 18]], axis=1)

        Arms = (L + R) / 2
        plot_list[0].set_data(time_x, T)
        plot_list[1].set_data(time_x, L)
        plot_list[2].set_data(time_x, R)

        plot_list[3].set_data(time_x, T)
        plot_list[4].set_data(time_x, Arms)

        time_x = np.arange(0, self.annotations_v1.annotations['Process'].shape[0])
        print(np.max(self.annotations_v1.annotations['Process']))
        plot_list[5].set_data(time_x, self.annotations_v1.annotations['Activity'] * 1000)
        plot_list[6].set_data(time_x, self.annotations_v1.annotations['Process'] * 1000)

        #self.annotations_v1.annotations['Process']

        plot_list[7].set_data(time_x, self.annotations_v1.annotations['Process'])
        plot_list[8].set_data(time_x, self.annotations_v1.annotations['Activity'])

        axis_list[0].relim()
        axis_list[0].autoscale_view()
        axis_list[0].legend(loc="best")

        axis_list[1].relim()
        axis_list[1].autoscale_view()
        axis_list[1].legend(loc="best")

        axis_list[2].relim()
        axis_list[2].autoscale_view()
        axis_list[2].legend(loc="best")

        fig.canvas.draw()
        plt.show()
        # plt.pause(2.0)

        return

if __name__ == "__main__":

    #projects = Preprocessing()
    #list_projects = projects.get_projects("/home/fernando/Documents/gpu_2_data/Documents/CAR/datasets.json")
    #print(projects.get_projects("/home/fernando/Documents/gpu_2_data/Documents/CAR/datasets.json"))

    #project = Project(list_projects[0], list_projects[1][0])

    #project.visualize()
    print("Done")