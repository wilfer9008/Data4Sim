"""
@created: 07.05.2024
@author: Fernando Moya Rueda

@copyright: Motion Miners GmbH, Emil-Figge Str. 76, 44227 Dortmund, 2022

@brief: For listing and preparing the CAR classes
"""
import json
import numpy as np
import csv

headers = [
            "CL114_Preparing_Order",
            "CL115_Picking_Travel_Time",
            "CL116_Picking_Pick_Time",
            "CL117_Unpacking",
            "CL118_Packing",
            "CL119_Storing_Travel_Time",
            "CL120_Storing_Store_Time",
            "CL121_Finalizing_Order",
            "CL122_Another_Mid_Level_Process",
            "CL123_Mid_Level_Process_Unknown",
            "CL124_Collecting_Order_and_Hardware",
            "CL125_Collecting_Cart",
            "CL126_Collecting_Empty_Cardboard_Boxes",
            "CL127_Collecting_Packed_Cardboard_Boxes",
            "CL128_Transporting_a_Cart_to_the_Base",
            "CL129_Transporting_to_the_Packaging_Sorting_Area",
            "CL130_Handing_Over_Packed_Cardboard_Boxes",
            "CL131_Returning_Empty_Cardboard_Boxes",
            "CL132_Returning_Cart",
            "CL133_Returning_Hardware",
            "CL134_Waiting",
            "CL135_Reporting_and_Clarifying_the_Incident",
            "CL136_Removing_Cardboard_Box_Item_from_the_Cart",
            "CL137_Moving_to_the_Next_Position",
            "CL138_Placing_Items_on_a_Rack",
            "CL139_Retrieving_Items",
            "CL140_Moving_to_a_Cart",
            "CL141_Placing_Cardboard_Box_Item_on_a_Table",
            "CL142_Opening_Cardboard_Box",
            "CL143_Disposing_of_Filling_Material_or_Shipping_Label",
            "CL144_Sorting",
            "CL145_Filling_Cardboard_Box_with_Filling_Material",
            "CL146_Printing_Shipping_Label_and_Return_Slip",
            "CL147_Preparing_or_Adding_Return_Label",
            "CL148_Attaching_Shipping_Label",
            "CL149_Removing_Elastic_Band",
            "CL150_Sealing_Cardboard_Box",
            "CL151_Placing_Cardboard_Box_Item_in_a_Cart",
            "CL152_Tying_Elastic_Band_Around_Cardboard",
            "CL153_Another_Low_Level_Process",
            "CL154_Low_Level_Process_Unknown",
            "CL001_Synchronization",
            "CL002_Confirming_with_Pen",
            "CL003_Confirming_with_Screen",
            "CL004_Confirming_with_Button",
            "CL005_Scanning",
            "CL006_Pulling_Cart",
            "CL007_Pushing_Cart",
            "CL008_Handling_Upwards",
            "CL009_Handling_Centered",
            "CL010_Handling_Downwards",
            "CL011_Walking",
            "CL012_Standing",
            "CL013_Sitting",
            "CL014_Another_Main_Activity",
            "CL015_Main_Activity_Unknown"
            ]

class Annotation(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.scheme_path = None
        self.scheme = None
        self.raw_data = None
        self.data = None
        self.raw_annotations = {}
        self.annotations = {}
        self.statistics = {}
        self.valid_range = [0, 0]
        self.valid_range_sensors = None
        self.generation = "gen5"

        return

    def set_scheme_path(self, path):
        self.scheme_path = path
        return

    def get_annotations_scheme(self):
        '''
        Lists the available annotated CAR annotations

        returns
        @param :
        '''

        # Opening JSON file
        f = open(self.scheme_path)

        self.scheme = json.load(f)
        self.scheme = dict(self.scheme)
        f.close()

        return self.scheme

    def set_annotations(self, data, path):
        self.raw_data = data

        if data.shape[1] != 56:
            print("ERROR: NOT UPDATED ANNOTATION FILE------- SIZE {}".format(data.shape))
        else:
            print("UPDATED ANNOTATION FILE------- SIZE {}".format(data.shape))

        self.raw_annotations['Mid_Process'] = np.argmax(self.raw_data[:, 0: 10], axis=1)
        self.raw_annotations['Low_Process'] = np.argmax(self.raw_data[:, 10: 41], axis=1)
        self.raw_annotations['Activity'] = self.raw_data[:, 41: 56]

        self.set_valid_range(path=path)

        print("Statistics:")
        print("Mid_Process: ", np.bincount(self.raw_annotations['Mid_Process'], minlength=10))
        print("Mid_Process Percentage: ", np.bincount(self.raw_annotations['Mid_Process'], minlength=10) / self.raw_annotations['Mid_Process'].shape[0] * 100)
        print(np.sum(np.sum(self.raw_data[:, 0: 10], axis=1)), " from ", self.raw_data.shape[0])
        if (np.sum(np.sum(self.raw_data[:, 0: 10], axis=1)) != self.raw_data.shape[0]):
            print("Error with the labels Process--------------------------->")
            print(np.where(np.sum(self.raw_data[:, 0: 10], axis=1) > 1))
            print(np.where(np.sum(self.raw_data[:, 0: 10], axis=1) == 0))
        print("Low_Process: ", np.bincount(self.raw_annotations['Low_Process'], minlength=31))
        print("Low_Process Percentage: ", np.bincount(self.raw_annotations['Low_Process'], minlength=31) / self.raw_annotations['Low_Process'].shape[0] * 100)
        print(np.sum(np.sum(self.raw_data[:, 10: 41], axis=1)), " from ", self.raw_data.shape[0])
        if (np.sum(np.sum(self.raw_data[:, 10: 41], axis=1)) != self.raw_data.shape[0]):
            print("Error with the labels Activities--------------------------->")
            print(np.where(np.sum(self.raw_data[:, 10: 41], axis=1) > 1))
            print(np.where(np.sum(self.raw_data[:, 10: 41], axis=1) == 0))

        return

    def set_sync_annotations(self, data):
        self.data = data

        self.annotations['Mid_Process'] = np.argmax(self.data[:, 0: 10], axis=1)
        self.annotations['Low_Process'] = np.argmax(self.data[:, 10: 41], axis=1)
        self.annotations['Activity'] = self.data[:, 41: 56]

        print("Statistics:")
        print("Mid_Process: ", np.bincount(self.annotations['Mid_Process'], minlength=9))
        print(np.sum(np.sum(self.data[:, 0: 10], axis=1)), " from ", self.data.shape[0])
        if (np.sum(np.sum(self.data[:, 0: 10], axis=1)) != self.data.shape[0]):
            print("Error with the labels--------------------------->")
        print("Low_Process: ", np.bincount(self.annotations['Low_Process'], minlength=14))
        print(np.sum(np.sum(self.data[:, 10: 41], axis=1)), " from ", self.data.shape[0])
        if (np.sum(np.sum(self.data[:, 10: 41], axis=1)) != self.data.shape[0]):
            print("Error with the labels--------------------------->")



        return


    def get_annotations(self):
        return self.annotations

    def get_statistics_annotations(self):

        self.statistics['Mid_Process'] = np.bincount(self.annotations['Mid_Process'], minlength=10)
        self.statistics['Low_Process'] = np.bincount(self.annotations['Low_Process'], minlength=31)
        self.statistics['Activity'] = np.sum(self.annotations['Activity'], axis=15)

        return self.statistics

    def set_valid_range(self, path):

        # Opening JSON file
        f = open(path + "project.json")
        ranges = json.load(f)
        ranges = dict(ranges)
        f.close()

        #self.valid_range[0] = int(-1 * (ranges[] / 1000 * 60))
        #self.valid_range[1] = np.where(self.raw_annotations['Process'] != 0)[0][-1]

        self.valid_range[0] = ranges["start"]
        self.valid_range[1] = ranges["end"]
        self.generation = ranges["generation"]

        if "start_sensors" in ranges.keys():
            self.valid_range_sensors = [ranges["start_sensors"], ranges["end_sensors"]]

        return

    def crop_annotations(self):
        self.data = self.raw_data[self.valid_range[0]: self.valid_range[1]]

        self.annotations['Mid_Process'] = np.argmax(self.data[:, 0: 10], axis=1)
        self.annotations['Low_Process'] = np.argmax(self.data[:, 10: 41], axis=1)
        self.annotations['Activity'] = self.data[:, 41: 56]

        return

    def sync_annotations(self, length_sensors):

        rate_time = (self.data.shape[0] / 60) / (((length_sensors * 10) / 1000)/60)
        annotation_time = np.arange(0, self.data.shape[0], rate_time / 100)

        if len(annotation_time) == length_sensors:
            self.data = self.data[annotation_time.astype(int)]
        else:
            self.data = self.data[annotation_time[:-1].astype(int)]

        self.annotations['Mid_Process'] = np.argmax(self.data[:, 0: 10], axis=1)
        self.annotations['Low_Process'] = np.argmax(self.data[:, 10: 41], axis=1)
        self.annotations['Activity'] = self.data[:, 41: 56]

        return

    def save_sync_annotations(self, filename):

        with open(filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(headers)
            for d in self.data:
                spamwriter.writerow(d)

        return

if __name__ == "__main__":

    annotation = Annotation()
    annotation.set_scheme_path("../scheme/scheme_V4.json")
    print(annotation.get_annotations_scheme())

    print("Done")