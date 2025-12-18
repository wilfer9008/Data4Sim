"""
@created: 14.05.2021
@author: Fernando Moya Rueda

@copyright: Motion Miners GmbH, Emil-Figge Str. 76, 44227 Dortmund, 2022

@brief: Preprocessing of ErgoCom sequences
        It will extract the recordings from OBDs, segment them, and prepare the
        sets for training, validation and testing
"""

import json
import os
import sys
import pickle
import numpy as np
from projects import Project
from sliding_window import sliding_window

class Preprocessing(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.root = path
        self.project_list = None
        self.get_projects()
        self.set_sets()
        return

    def get_projects(self):
        # Opening JSON file
        f = open(self.root + "datasets.json")
        self.project_list = json.load(f)
        f.close()

        return

    def set_sets(self):

        print(self.project_list['projects'])
        number_subjects = len(self.project_list['projects'])

        # Partitions of training validation and testing proportions
        tr_pr = int(number_subjects * 0.7)
        val_pr = int(number_subjects * 0.85)
        test_pr = int(number_subjects * 1.0)

        self.train_ids = self.project_list['projects'][:tr_pr]
        self.train_final_ids = self.project_list['projects'][: val_pr]
        #self.train_final_ids = self.train_ids
        self.val_ids = self.project_list['projects'][tr_pr: val_pr]
        #self.val_ids = self.train_ids
        self.test_ids = self.project_list['projects'][val_pr: test_pr]

        return

    def opp_sliding_window(self, data_x, data_y, ws, ss):
        """
        Performs the sliding window approach on the data and the labels

        return three arrays.
        - data, an array where first dim is the windows
        - labels per window according to end, middle or mode
        - all labels per window

        @param data_x: ids for train
        @param data_y: ids for train
        @param ws: ids for train
        @param ss: ids for train
        @param label_pos_end: ids for train
        """

        print("Sliding window: Creating windows {} with step {}".format(ws, ss))

        data_x_segments = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
        # All labels per window
        #data_y_all = np.asarray([i[:] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
        data_y_all = sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))

        return data_x_segments.astype(np.float16), data_y_all.astype(np.uint8)


    def get_unique_attrs(self, ids, data_dir=None, usage_modus="train"):
        """
        creates files for each of the sequences extracted from a file
        following a sliding window approach

        returns a numpy array

        @param ids: list of recordings for the given subset
        @param base_path: location of raw data
        @param sliding_window_length: length of window for segmentation
        @param sliding_window_step: step between windows for segmentation
        @param data_dir: path to dir where files will be stored
        @param usage_modus: type of subset (train, val, test)
        """

        unique_attrs = np.empty(shape = (0, 85))
        # Iterates over the recordings
        print(usage_modus, " modus with IDS:", ids)
        for P in ids:
            try:
                project = Project(self.project_list['root'], P)
                project.load_annotated_sync_data()
                print("\n{}\n{}".format(project.root, project.project_path))

                try:
                    # Getting labels and attributes
                    labels = project.annotations_v1.data
                    labels_process = project.annotations_v1.annotations['Process']
                    labels_activities = project.annotations_v1.annotations['Activity']
                    print("\nLabels loaded in modus {} with shape {}".format(usage_modus, labels.shape))

                    # Deleting rows containing the "Null" class if wanted
                    class_labels_null = np.where(labels_process == 0)[0]
                    class_labels_error = np.where(labels_process == 8)[0]
                    class_labels_ignore = np.where(labels_activities == 0)[0]
                    class_labels = np.unique(np.concatenate((class_labels_null, class_labels_error, class_labels_ignore)))
                    labels = np.delete(labels, class_labels, 0)
                    print("\nLabel Null, Error, Ignore eliminated {}".format(labels.shape))

                    class_labels = [0, 8, 9]
                    labels = np.delete(labels, class_labels, 1)

                    labels = np.unique(labels, axis=0)
                    unique_attrs = np.concatenate((unique_attrs, labels), axis=0)
                    print("\nLabel attrs eliminated {}".format(labels.shape))

                except KeyboardInterrupt:
                    print("\nYou cancelled the operation.")
                    break
                except Exception as e:
                    print("\nError {}.".format(e))
                    break
                except:
                    print("2 In generating data, Error getting the labels {}".format(P))
                    continue

            except KeyboardInterrupt:
                print("\nYou cancelled the operation.")
                break
            except Exception as e:
                print("\nError {}.".format(e))
                break

        unique_attrs_total = np.unique(unique_attrs, axis=0)

        processes = np.argmax(unique_attrs_total[:, :7], axis=1)
        activities = np.argmax(unique_attrs_total[:, 7:19], axis=1)
        attrs = unique_attrs_total[:, 19:]

        attr_rep_processes = np.zeros(shape=(unique_attrs_total.shape[0], 62))
        attr_rep_processes[:, 0] = processes
        attr_rep_processes[:, 1:] = unique_attrs_total[:, 7:]

        attr_rep_activities = np.zeros(shape=(unique_attrs_total.shape[0], 50))
        attr_rep_activities[:, 0] = activities
        attr_rep_activities[:, 1:] = attrs

        np.savetxt(self.root + "Segmented_windows/" + self.project_list[
            'name'] + "/" + usage_modus + "_attrs_rep_processes.csv", attr_rep_processes, delimiter="\n", fmt="%s")

        np.savetxt(self.root + "Segmented_windows/" + self.project_list[
            'name'] + "/" + usage_modus + "_attr_rep_activities.csv", attr_rep_activities, delimiter="\n", fmt="%s")

        return


    def generate_CSV(self, csv_dir, type_subset, data_dir):
        """
        Creates a CSV file with the path of all the segmented windows from a given subset

        @param csv_dir: file name
        @param type_subset: type of subset ("train.csv", "val.csv", "test.csv")
        @param data_dir: path to the segmented windows of the given type subset

        @return: f: list with all the segmented-windows paths
        """
        f = []
        for dirpath, dirnames, filenames in os.walk(data_dir):
            for n in range(len(filenames)):
                f.append(data_dir + "seq_{0:07}.pkl".format(n))

        np.savetxt(csv_dir + type_subset, f, delimiter="\n", fmt="%s")

        return f

    def generate_CSV_final(self, csv_dir, data_dir1, data_dir2):
        """
        Creates a CSV file with the path of all the segmented windows from training and validation sets

        @param csv_dir: file name
        @param data_dir1: path to the segmented windows of the training set
        @param data_dir2: path to the segmented windows of the validaiton set

        @return: f: list with all the segmented-windows paths
        """
        f = []
        for dirpath, dirnames, filenames in os.walk(data_dir1):
            for n in range(len(filenames)):
                f.append(data_dir1 + "seq_{0:07}.pkl".format(n))

        for dirpath, dirnames, filenames in os.walk(data_dir2):
            for n in range(len(filenames)):
                f.append(data_dir2 + "seq_{0:07}.pkl".format(n))

        np.savetxt(csv_dir, f, delimiter="\n", fmt="%s")

        return f

    def generate_data(self, ids, data_dir=None, usage_modus="train"):
        """
        creates files for each of the sequences extracted from a file
        following a sliding window approach

        returns a numpy array

        @param ids: list of recordings for the given subset
        @param base_path: location of raw data
        @param sliding_window_length: length of window for segmentation
        @param sliding_window_step: step between windows for segmentation
        @param data_dir: path to dir where files will be stored
        @param usage_modus: type of subset (train, val, test)
        """

        counter_seq = 0
        counter_file_label = -1
        hist_classes_all = np.zeros((8))
        hist_classes_acts_all = np.zeros((14))

        # Iterates over the recordings
        print(usage_modus, " modus with IDS:", ids)
        for P in ids:
            try:
                project = Project(self.project_list['root'], P)
                project.load_annotated_sync_data()
                print("\n{}\n{}".format(project.root, project.project_path))
                try:
                    # getting raw data
                    data = project.raw_data
                    data_x = data[:, 1:]
                    print("\nFiles loaded in modus {}\n{}".format(usage_modus, data.shape))
                except:
                    print("\n1 In loading data,  in file {}".format(P))
                    continue

                try:
                    # Getting labels and attributes
                    labels = project.annotations_v1.data
                    if labels.shape[1] != 86:
                        print("ERROR: NOT UPDATED ANNOTATION FILE------- SIZE {}".format(labels.shape))
                    else:
                        print("UPDATED ANNOTATION FILE------- SIZE {}".format(labels.shape))

                    labels_process = project.annotations_v1.annotations['Process']
                    labels_activities = project.annotations_v1.annotations['Activity']
                    print("\nLabels loaded in modus {} with shape {}".format(usage_modus, labels.shape))

                    # Deleting rows containing the "Null" class if wanted
                    class_labels_null = np.where(labels_process == 0)[0]
                    class_labels_error = np.where(labels_process == 8)[0]
                    class_labels_ignore = np.where(labels_activities == 0)[0]
                    #class_labels = np.unique(np.concatenate((class_labels_null, class_labels_error)))
                    #class_labels = np.unique(np.concatenate((class_labels_null, class_labels_error, class_labels_ignore)))
                    #data_x = np.delete(data_x, class_labels_error, 0)
                    #labels = np.delete(labels, class_labels_error, 0)
                    print("\nLabel Null, Error, Ignore eliminated {}".format(labels.shape))

                    # Deleting rows containing the "Error" class if wanted
                    #class_labels = np.where(labels_process == 8)[0]
                    #data_x = np.delete(data_x, class_labels, 0)
                    #labels = np.delete(labels, class_labels, 0)
                    #print("\nLabel Error eliminated {}".format(labels.shape))

                    # Deleting rows containing the "ignore" class
                    #class_labels = np.where(labels_activities == 0)[0]
                    #data_x = np.delete(data_x, class_labels, 0)
                    #labels = np.delete(labels, class_labels, 0)
                    #print("\nLabel Ignore eliminated {}".format(labels.shape))


                    null_elements = labels[:, 0] * 0
                    #class_none_ignore_labels = np.unique(np.concatenate((class_labels_null, class_labels_ignore)))
                    null_elements[class_labels_null] = 1
                    labels[:, 0] = null_elements
                    labels[:, 9] = null_elements
                    class_labels = [8]
                    labels = np.delete(labels, class_labels, 1)

                    data_x = np.delete(data_x, class_labels_error, 0)
                    labels = np.delete(labels, class_labels_error, 0)
                    print("\nLabel attrs eliminated {}".format(labels.shape))

                except KeyboardInterrupt:
                    print("\nYou cancelled the operation.")
                    break
                except Exception as e:
                    print("\nError {}.".format(e))
                    break
                except:
                    print("2 In generating data, Error getting the labels {}".format(P))
                    continue

                try:
                    # Sliding window approach
                    print("\nStarting sliding window")
                    X, Y = self.opp_sliding_window(
                        data_x, labels.astype(int),
                        self.project_list["sliding_window_length"],
                        self.project_list["sliding_window_step"]
                    )
                    print("\nWindows are extracted")

                    # Statistics

                    hist_classes = np.bincount(np.argmax(Y.reshape((-1, 85))[:, 0:8], axis=1), minlength=8)
                    hist_classes_all += hist_classes
                    print("\nNumber of seq per process {}".format(hist_classes))
                    print("Total Number of seq per process {}".format(hist_classes_all))

                    hist_classes_acts = np.bincount(np.argmax(Y.reshape((-1, 85))[:, 8:22], axis=1), minlength=14)
                    hist_classes_acts_all += hist_classes_acts
                    print("hist_classes_acts", hist_classes_acts.shape)
                    print("hist_classes_acts_all", hist_classes_acts_all.shape)
                    print("Y", Y.shape)
                    print("Y.reshape((-1, 85))[:, 8:22]", Y.reshape((-1, 85))[:, 8:22].shape)
                    print("\nNumber of seq per activity {}".format(hist_classes_acts))
                    print("Total Number of seq per activity {}".format(hist_classes_acts_all))


                    counter_file_label += 1

                    for f in range(X.shape[0]):
                        try:
                            # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                            seq = np.reshape(X[f], newshape=(1, X.shape[1], X.shape[2]))
                            seq = np.require(seq, dtype=np.float16)

                            obj = {
                                "data": seq,
                                "labels": Y[f],
                                "identity": P,
                                "label_file": counter_file_label,
                            }
                            file_name = open(os.path.join(data_dir, "seq_{0:07}.pkl".format(counter_seq)), "wb")

                            pickle.dump(obj, file_name, protocol=pickle.HIGHEST_PROTOCOL)
                            counter_seq += 1

                            sys.stdout.write(
                                "\r" + "Creating sequence file number {} with id {}".format(f, counter_seq)
                            )
                            sys.stdout.flush()
                            file_name.close()

                        except KeyboardInterrupt:
                            print("\nYou cancelled the operation.")
                            break
                        except Exception as e:
                            print("\nError {}.".format(e))
                            break
                        except:
                            raise ("\nError adding the seq {} from {} \n".format(f, X.shape[0]))

                    print("\nCorrect data extraction from {}".format(P))

                    del data
                    del data_x
                    del X
                    del labels
                    del class_labels

                except KeyboardInterrupt:
                    print("\nYou cancelled the operation.")
                    break
                except Exception as e:
                    print("\nError {}.".format(e))
                    break
                except:
                    print("\n5 In generating data, No created file {}".format(P))
            except KeyboardInterrupt:
                print("\nYou cancelled the operation.")
                break
            except Exception as e:
                print("\nError {}.".format(e))
                break

        print("\nFinal Number of seq per class {}".format(hist_classes_all))
        print("Final Number of files {}".format(counter_file_label))

        np.savetxt(self.root + "Segmented_windows/" + self.project_list[
            'name'] + "/" + usage_modus + "_process_statistics.csv", hist_classes_all, delimiter="\n", fmt="%s")

        np.savetxt(self.root + "Segmented_windows/" + self.project_list[
            'name'] + "/" + usage_modus + "_activities_statistics.csv", hist_classes_acts_all, delimiter="\n", fmt="%s")

        return counter_file_label

    def create_dataset(self):
        """
        Creates the dataset to be used for training. Windows of size [window_size, number of channels] are created for
        each of the recordings. Segmented windows are stored in a given path in pickle files. Segmented windows are divided
        in training, validation and testing sets, which are stored in given paths for each set. Besides, it creates
        CSV files containing the path of each segmented window (This to be used for the DataLoader)

        @param recordings: list of recordings
        @param base_path: base path where raw recordings are stored. These shall be already annotated and organised in
                        "_data.csv" and "_labels.csv"
        @return:
        """


        base_directory = self.root + "Segmented_windows/" + self.project_list['name'] + "/"

        data_dir_train = base_directory + "sequences_train/"
        if not os.path.exists(data_dir_train):
            os.makedirs(data_dir_train)

        data_dir_val = base_directory + "sequences_val/"
        if not os.path.exists(data_dir_val):
            os.makedirs(data_dir_val)

        data_dir_test = base_directory + "sequences_test_BBr_MS1_P1-1/"
        if not os.path.exists(data_dir_test):
            os.makedirs(data_dir_test)

        '''
        counter_file_label = self.generate_data(
            self.train_ids,
            data_dir=data_dir_train,
            usage_modus="train",
        )
        self.generate_data(
            self.val_ids,
            data_dir=data_dir_val,
            usage_modus="val"
        )
        '''

        self.generate_data(
            self.test_ids,
            data_dir=data_dir_test,
            usage_modus="test",
        )


        #self.get_unique_attrs(self.train_final_ids, data_dir=data_dir_train, usage_modus="train")

        #self.generate_CSV(base_directory, "train.csv", data_dir_train)
        #self.generate_CSV(base_directory, "val.csv", data_dir_val)
        self.generate_CSV(base_directory, "test_BBr_MS1_P1-1.csv", data_dir_test)
        #self.generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)

        #print("number of segmented windows of the training set {}".format(counter_file_label))

        return

if __name__ == "__main__":
    path = "/mnt/data/femo/Documents/CAR/"
    #path = "/home/fernando/Documents/gpu_2_data/Documents/CAR/"

    preprocessing = Preprocessing(path)
    preprocessing.create_dataset()
    #create_dataset_ergocom(recordings, path_2_extracted_recordings + "Extracted_Recordings/")

    print("Done")