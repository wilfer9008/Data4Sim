"""
@created: 16.03.2021
@author: Fernando Moya Rueda

@copyright: Motion Miners GmbH, Emil-Figge Str. 76, 44227 Dortmund, 2022

@brief: Training, Validaiton and Testing of the Network
"""

import logging
import math
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from HARWindows import HARWindows
from metrics import Metrics
from network import Network
from freezing_network_sections import Freezing

# from sacred import Experiment

range_scores = {}
range_scores["Process"] = np.arange(0, 8)
range_scores["Activity"] = np.arange(8, 22)
range_scores["Short Activity"] = np.arange(22, 45)
range_scores["Pose"] = np.arange(45, 79)
range_scores["temporal_group"] = np.arange(79, 85)
range_scores["attrs"] = np.arange(22, 85)

class Network_User(object):
    """
    classdocs
    """

    def __init__(self, config, exp):
        """
        Constructor
        """

        logging.info("        Network_User: Constructor")

        self.config = config
        self.device = torch.device("cuda:{}".format(self.config["GPU"]) if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.attrs = self.reader_att_rep("atts_per_class_mm_car.txt")
        self.attr_representation = self.reader_att_rep("atts_per_class_mm_car.txt")

        self.normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.001]))
        self.normal_mult = torch.distributions.Normal(torch.tensor([1.0]), torch.tensor([0.001]))
        self.exp = exp

        return

    ##################################################
    ###################  reader_att_rep  #############
    ##################################################

    def reader_att_rep(self, path: str) -> np.array:
        """
        gets attribute representation from txt file.

        returns a numpy array

        @param path: path to file
        @param att_rep: Numpy matrix with the attribute representation
        """

        att_rep = np.loadtxt(path, delimiter=",", skiprows=0)
        return att_rep

    def setting_samples_from_windows(self, targets, predictions_test, targets_files):
        logging.info(
            "        Network_User:            Segmentation:    "
            "with these number of files {}".format(int(torch.max(targets_files).item()))
        )

        for target_files in range(0, 1 + int(torch.max(targets_files).item())):
            targets_per_file = targets[targets_files == target_files]
            predictions_per_file = predictions_test[targets_files == target_files]
            size_samples = (
                self.config["sliding_window_step"] * (targets_per_file.size(0) - 1)
                + self.config["sliding_window_length"]
            )
            sample_targets = torch.zeros((self.config["num_classes"], size_samples)).to(self.device, dtype=torch.long)
            sample_predictions = torch.zeros((self.config["num_classes"], size_samples)).to(
                self.device, dtype=torch.long
            )
            for ws in range(targets_per_file.size(0)):
                window_samples = targets_per_file[ws]
                window_samples = nn.functional.one_hot(
                    window_samples.type(torch.LongTensor), num_classes=self.config["num_classes"]
                )
                window_samples = torch.transpose(window_samples, 0, 1).to(self.device, dtype=torch.long)
                sample_targets[
                    :,
                    self.config["sliding_window_step"] * ws : (self.config["sliding_window_step"] * ws)
                    + self.config["sliding_window_length"],
                ] += window_samples

                if self.config["aggregate"] in ["FCN", "LSTM"]:
                    window_samples = predictions_per_file[ws]
                else:
                    window_samples = torch.ones((targets_per_file[ws].size())).to(self.device, dtype=torch.long)
                    window_samples = window_samples * predictions_per_file[ws]
                window_samples = nn.functional.one_hot(
                    window_samples.type(torch.LongTensor), num_classes=self.config["num_classes"]
                )
                window_samples = torch.transpose(window_samples, 0, 1).to(self.device, dtype=torch.long)
                sample_predictions[
                    :,
                    self.config["sliding_window_step"] * ws : (self.config["sliding_window_step"] * ws)
                    + self.config["sliding_window_length"],
                ] += window_samples

            # sample_targets_single = torch.zeros(size_samples)
            # sample_predictions_single = torch.zeros(size_samples)
            # for ws in range(size_samples):
            # bincounts = torch.bincount(sample_targets[:, ws].type(dtype=torch.long),
            #                           minlength=((self.config["num_classes"] + 1)))
            # bincounts, bincounts_idx = torch.sort(bincounts)
            sample_targets_single = torch.argmax(sample_targets.type(dtype=torch.long), axis=0)

            # bincounts = torch.bincount(sample_predictions[:, ws].type(dtype=torch.long),
            #                           minlength=(self.config["num_classes"] + 1))
            # bincounts, bincounts_idx = torch.sort(bincounts)
            sample_predictions_single = torch.argmax(sample_predictions.type(dtype=torch.long), axis=0)

            if target_files == 0:
                sample_targets_single_files = sample_targets_single
            else:
                sample_targets_single_files = torch.cat((sample_targets_single_files, sample_targets_single), dim=0)

            if target_files == 0:
                sample_predictions_single_files = sample_predictions_single
            else:
                sample_predictions_single_files = torch.cat(
                    (sample_predictions_single_files, sample_predictions_single), dim=0
                )

        logging.info(
            "        Network_User:            Segmentation:    "
            "size of sequence labels {}".format(sample_targets_single_files.size())
        )

        # del sample_targets_single, sample_predictions_single
        # del targets_per_file, predictions_per_file

        return sample_targets_single_files, sample_predictions_single_files

    ##################################################
    ################  load_weights  ##################
    ##################################################

    def load_weights(self, network):
        """
        Load weights from a trained network

        @param network: target network with orthonormal initialisation
        @return network: network with transferred CNN layers
        """
        model_dict = network.state_dict()
        # 1. filter out unnecessary keys
        logging.info("        Network_User:        Loading Weights")

        if self.config["pretrained"]:
            pretrained_dict = torch.load(self.config["folder_exp_base_fine_tuning"] + "old_network_context.pt",
                                         map_location=torch.device('cpu'))["state_dict"]

            logging.info("        Network_User:        Pretrained model loaded")

            # for k, v in pretrained_dict.items():
            #    print(k)

            if self.config["network"] == "cnn_imu":
                list_layers = [
                    "conv_LA_1_1.weight",
                    "conv_LA_1_1.bias",
                    "conv_LA_1_2.weight",
                    "conv_LA_1_2.bias",
                    "conv_LA_2_1.weight",
                    "conv_LA_2_1.bias",
                    "conv_LA_2_2.weight",
                    "conv_LA_2_2.bias",
                    "fc3_LA.weight",
                    "fc3_LA.bias",
                    "conv_N_1_1.weight",
                    "conv_N_1_1.bias",
                    "conv_N_1_2.weight",
                    "conv_N_1_2.bias",
                    "conv_N_2_1.weight",
                    "conv_N_2_1.bias",
                    "conv_N_2_2.weight",
                    "conv_N_2_2.bias",
                    "fc3_N.weight",
                    "fc3_N.bias",
                    "conv_RA_1_1.weight",
                    "conv_RA_1_1.bias",
                    "conv_RA_1_2.weight",
                    "conv_RA_1_2.bias",
                    "conv_RA_2_1.weight",
                    "conv_RA_2_1.bias",
                    "conv_RA_2_2.weight",
                    "conv_RA_2_2.bias",
                    "fc3_RA.weight",
                    "fc3_RA.bias",
                    "conv_LL_1_1.weight",
                    "conv_LL_1_1.bias",
                    "conv_LL_1_2.weight",
                    "conv_LL_1_2.bias",
                    "conv_LL_2_1.weight",
                    "conv_LL_2_1.bias",
                    "conv_LL_2_2.weight",
                    "conv_LL_2_2.bias",
                    "conv_RL_1_1.weight",
                    "conv_RL_1_1.bias",
                    "conv_RL_1_2.weight",
                    "conv_RL_1_2.bias",
                    "conv_RL_2_1.weight",
                    "conv_RL_2_1.bias",
                    "conv_RL_2_2.weight",
                    "conv_RL_2_2.bias",
                    "fc4.weight",
                    "fc4.bias",
                    "BASE_ACTIVITIES.weight",
                    "BASE_ACTIVITIES.bias",
                    "HANDLING_HEIGHTS.weight",
                    "HANDLING_HEIGHTS.bias",
                    "WRAP.weight",
                    "WRAP.bias",
                    "LARA.weight",
                    "LARA.bias",
                    "PULL_PUSH.weight",
                    "PULL_PUSH.bias",
                    "WRITE.weight",
                    "WRITE.bias",
                    "AIRBUS.weight",
                    "AIRBUS.bias",
                ]
        else:
            if self.config["pooling"] in [1, 3]:
                pretrained_dict = torch.load(self.config["folder_exp_base_fine_tuning"] + "network_pool1.pt")["state_dict"]
            elif self.config["pooling"] in [2, 4]:
                pretrained_dict = torch.load(self.config["folder_exp_base_fine_tuning"] + "network_pool2.pt")["state_dict"]
            else:
                pretrained_dict = torch.load(self.config["folder_exp_base_fine_tuning"] + "network.pt")["state_dict"]
            logging.info("        Network_User:        Pretrained model loaded")

            # for k, v in pretrained_dict.items():
            #    print(k)

            if self.config["network"] == "cnn_imu":
                list_layers = [
                    "conv_LA_1_1.weight",
                    "conv_LA_1_1.bias",
                    "conv_LA_1_2.weight",
                    "conv_LA_1_2.bias",
                    "conv_LA_2_1.weight",
                    "conv_LA_2_1.bias",
                    "conv_LA_2_2.weight",
                    "conv_LA_2_2.bias",
                    "fc3_LA.weight",
                    "fc3_LA.bias",
                    "conv_N_1_1.weight",
                    "conv_N_1_1.bias",
                    "conv_N_1_2.weight",
                    "conv_N_1_2.bias",
                    "conv_N_2_1.weight",
                    "conv_N_2_1.bias",
                    "conv_N_2_2.weight",
                    "conv_N_2_2.bias",
                    "fc3_N.weight",
                    "fc3_N.bias",
                    "conv_RA_1_1.weight",
                    "conv_RA_1_1.bias",
                    "conv_RA_1_2.weight",
                    "conv_RA_1_2.bias",
                    "conv_RA_2_1.weight",
                    "conv_RA_2_1.bias",
                    "conv_RA_2_2.weight",
                    "conv_RA_2_2.bias",
                    "fc3_RA.weight",
                    "fc3_RA.bias",
                    "conv_LL_1_1.weight",
                    "conv_LL_1_1.bias",
                    "conv_LL_1_2.weight",
                    "conv_LL_1_2.bias",
                    "conv_LL_2_1.weight",
                    "conv_LL_2_1.bias",
                    "conv_LL_2_2.weight",
                    "conv_LL_2_2.bias",
                    "conv_RL_1_1.weight",
                    "conv_RL_1_1.bias",
                    "conv_RL_1_2.weight",
                    "conv_RL_1_2.bias",
                    "conv_RL_2_1.weight",
                    "conv_RL_2_1.bias",
                    "conv_RL_2_2.weight",
                    "conv_RL_2_2.bias",
                    "fc4.weight",
                    "fc4.bias",
                    "BASE_ACTIVITIES.weight",
                    "BASE_ACTIVITIES.bias",
                    "HANDLING_HEIGHTS.weight",
                    "HANDLING_HEIGHTS.bias",
                    "WRAP.weight",
                    "WRAP.bias",
                    "LARA.weight",
                    "LARA.bias",
                    "PULL_PUSH.weight",
                    "PULL_PUSH.bias",
                    "WRITE.weight",
                    "WRITE.bias",
                    "AIRBUS.weight",
                    "AIRBUS.bias",
                    "Trans_conv_LA_1_1.weight",
                    "Trans_conv_LA_1_1.bias",
                    "Trans_conv_LA_1_2.weight",
                    "Trans_conv_LA_1_2.bias",
                    "Trans_conv_LA_2_1.weight",
                    "Trans_conv_LA_2_1.bias",
                    "Trans_conv_LA_2_2.weight",
                    "Trans_conv_LA_2_2.bias",
                    "Trans_fc3_LA.weight",
                    "Trans_fc3_LA.bias",
                    "Trans_conv_N_1_1.weight",
                    "Trans_conv_N_1_1.bias",
                    "Trans_conv_N_1_2.weight",
                    "Trans_conv_N_1_2.bias",
                    "Trans_conv_N_2_1.weight",
                    "Trans_conv_N_2_1.bias",
                    "Trans_conv_N_2_2.weight",
                    "Trans_conv_N_2_2.bias",
                    "Trans_fc3_N.weight",
                    "Trans_fc3_N.bias",
                    "Trans_conv_RA_1_1.weight",
                    "Trans_conv_RA_1_1.bias",
                    "Trans_conv_RA_1_2.weight",
                    "Trans_conv_RA_1_2.bias",
                    "Trans_conv_RA_2_1.weight",
                    "Trans_conv_RA_2_1.bias",
                    "Trans_conv_RA_2_2.weight",
                    "Trans_conv_RA_2_2.bias",
                    "Trans_fc3_RA.weight",
                    "Trans_fc3_RA.bias",
                    "Trans_conv_LL_1_1.weight",
                    "Trans_conv_LL_1_1.bias",
                    "Trans_conv_LL_1_2.weight",
                    "Trans_conv_LL_1_2.bias",
                    "Trans_conv_LL_2_1.weight",
                    "Trans_conv_LL_2_1.bias",
                    "Trans_conv_LL_2_2.weight",
                    "Trans_conv_LL_2_2.bias",
                    "Trans_conv_RL_1_1.weight",
                    "Trans_conv_RL_1_1.bias",
                    "Trans_conv_RL_1_2.weight",
                    "Trans_conv_RL_1_2.bias",
                    "Trans_conv_RL_2_1.weight",
                    "Trans_conv_RL_2_1.bias",
                    "Trans_conv_RL_2_2.weight",
                    "Trans_conv_RL_2_2.bias",
                    "Trans_fc4.weight",
                    "Trans_fc4.bias",
                    "fc5.weight",
                    "fc5.bias",
                    "fc5_old_windows.weight",
                    "fc5_old_windows.bias",
                    "fc4_R.weight_ih_l0",
                    "fc4_R.weight_hh_l0",
                    "fc4_R.bias_ih_l0",
                    "fc4_R.bias_hh_l0",
                    "Trans_fc4_2.weight",
                    "Trans_fc4_2.bias",
                    "fc5_activity.weight",
                    "fc5_activity.bias",
                    "fc5_attrs_activity.weight",
                    "fc5_attrs_activity.bias",
                    "PROCESS.weight",
                    "PROCESS.bias",
                    "ACTIVITY.weight",
                    "ACTIVITY.bias",
                    "Trans_fc5.weight",
                    "Trans_fc5.bias",
                ]

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in list_layers}
        # print(pretrained_dict)

        logging.info("        Network_User:        Pretrained layers selected")
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        logging.info("        Network_User:        Pretrained layers selected")
        # 3. load the new state dict
        network.load_state_dict(model_dict)
        logging.info("        Network_User:        Weights loaded")

        return network

    ##################################################
    ############  set_required_grad  #################
    ##################################################

    def set_required_grad(self, network):
        """
        Setting the computing of the gradients for some layers as False
        This will act as the freezing of layers

        @param network: target network
        @return network: network with frozen layers
        """
        model_dict = network.state_dict()
        # 1. filter out unnecessary keys
        logging.info("        Network_User:        Setting Required_grad to Weights")


        if self.config["network"] == "cnn_imu":
            list_layers = [
                "conv_LA_1_1.weight",
                "conv_LA_1_1.bias",
                "conv_LA_1_2.weight",
                "conv_LA_1_2.bias",
                "conv_LA_2_1.weight",
                "conv_LA_2_1.bias",
                "conv_LA_2_2.weight",
                "conv_LA_2_2.bias",
                "conv_LL_1_1.weight",
                "conv_LL_1_1.bias",
                "conv_LL_1_2.weight",
                "conv_LL_1_2.bias",
                "conv_LL_2_1.weight",
                "conv_LL_2_1.bias",
                "conv_LL_2_2.weight",
                "conv_LL_2_2.bias",
                "conv_N_1_1.weight",
                "conv_N_1_1.bias",
                "conv_N_1_2.weight",
                "conv_N_1_2.bias",
                "conv_N_2_1.weight",
                "conv_N_2_1.bias",
                "conv_N_2_2.weight",
                "conv_N_2_2.bias",
                "conv_RA_1_1.weight",
                "conv_RA_1_1.bias",
                "conv_RA_1_2.weight",
                "conv_RA_1_2.bias",
                "conv_RA_2_1.weight",
                "conv_RA_2_1.bias",
                "conv_RA_2_2.weight",
                "conv_RA_2_2.bias",
                "conv_RL_1_1.weight",
                "conv_RL_1_1.bias",
                "conv_RL_1_2.weight",
                "conv_RL_1_2.bias",
                "conv_RL_2_1.weight",
                "conv_RL_2_1.bias",
                "conv_RL_2_2.weight",
                "conv_RL_2_2.bias",
                "fc3_LA.weight",
                "fc3_LA.bias",
                "fc3_LL.weight",
                "fc3_LL.bias",
                "fc3_N.weight",
                "fc3_N.bias",
                "fc3_RA.weight",
                "fc3_RA.bias",
                "fc3_RL.weight",
                "fc3_RL.bias",
                "fc4.weight",
                "fc4.bias",
                #"fc5.weight",
                #"fc5.bias",
                #"fc4_R.weight_ih_l0",
                #"fc4_R.weight_hh_l0",
                #"fc4_R.bias_ih_l0",
                #"fc4_R.bias_hh_l0",
                "BASE_ACTIVITIES.weight",
                "BASE_ACTIVITIES.bias",
                "HANDLING_HEIGHTS.weight",
                "HANDLING_HEIGHTS.bias",
                "WRAP.weight",
                "WRAP.bias",
                "LARA.weight",
                "LARA.bias",
                "PULL_PUSH.weight",
                "PULL_PUSH.bias",
                "WRITE.weight",
                "WRITE.bias",
                "AIRBUS.weight",
                "AIRBUS.bias",
            ]
            if self.config["dataset"] in [
                "motionminers_flw100",
                "ergocom_fernando",
                "ergocom_fernando150",
                "motionminers_flw150",
            ]:
                list_layers.remove("LARA.weight")
                list_layers.remove("LARA.bias")
                list_layers.remove("fc5.weight")
                list_layers.remove("fc5.bias")
            if self.config["dataset"] in ["motionminers_real", "motionminers_real150"]:
                list_layers.remove("BASE_ACTIVITIES.weight")
                list_layers.remove("BASE_ACTIVITIES.bias")
                list_layers.remove("fc5.weight")
                list_layers.remove("fc5.bias")

        for pn, pv in network.named_parameters():
            if pn in list_layers:
                pv.requires_grad = False

        return network


    ##################################################
    ############  set_required_grad  #################
    ##################################################

    def set_required_grad_branches_on(self, network):
        """
        Setting the computing of the gradients for some layers as False
        This will act as the freezing of layers

        @param network: target network
        @return network: network with frozen layers
        """
        model_dict = network.state_dict()
        # 1. filter out unnecessary keys
        logging.info("        Network_User:        Setting Required_grad to Weights")

        if self.config["network"] == "cnn_imu":
            list_layers = [
                "conv_LA_1_1.weight",
                "conv_LA_1_1.bias",
                "conv_LA_1_2.weight",
                "conv_LA_1_2.bias",
                "conv_LA_2_1.weight",
                "conv_LA_2_1.bias",
                "conv_LA_2_2.weight",
                "conv_LA_2_2.bias",
                "conv_LL_1_1.weight",
                "conv_LL_1_1.bias",
                "conv_LL_1_2.weight",
                "conv_LL_1_2.bias",
                "conv_LL_2_1.weight",
                "conv_LL_2_1.bias",
                "conv_LL_2_2.weight",
                "conv_LL_2_2.bias",
                "conv_N_1_1.weight",
                "conv_N_1_1.bias",
                "conv_N_1_2.weight",
                "conv_N_1_2.bias",
                "conv_N_2_1.weight",
                "conv_N_2_1.bias",
                "conv_N_2_2.weight",
                "conv_N_2_2.bias",
                "conv_RA_1_1.weight",
                "conv_RA_1_1.bias",
                "conv_RA_1_2.weight",
                "conv_RA_1_2.bias",
                "conv_RA_2_1.weight",
                "conv_RA_2_1.bias",
                "conv_RA_2_2.weight",
                "conv_RA_2_2.bias",
                "conv_RL_1_1.weight",
                "conv_RL_1_1.bias",
                "conv_RL_1_2.weight",
                "conv_RL_1_2.bias",
                "conv_RL_2_1.weight",
                "conv_RL_2_1.bias",
                "conv_RL_2_2.weight",
                "conv_RL_2_2.bias",
                "fc3_LA.weight",
                "fc3_LA.bias",
                "fc3_N.weight",
                "fc3_N.bias",
                "fc3_RA.weight",
                "fc3_RA.bias",
                "fc4.weight",
                "fc4.bias",
                "BASE_ACTIVITIES.weight",
                "BASE_ACTIVITIES.bias",
                "HANDLING_HEIGHTS.weight",
                "HANDLING_HEIGHTS.bias",
                "WRAP.weight",
                "WRAP.bias",
                "LARA.weight",
                "LARA.bias",
                "PULL_PUSH.weight",
                "PULL_PUSH.bias",
                "WRITE.weight",
                "WRITE.bias",
                "AIRBUS.weight",
                "AIRBUS.bias",
            ]

        for pn, pv in network.named_parameters():
            if pn in list_layers:
                pv.requires_grad = False

        return network

    ##################################################
    ###################  train  ######################
    ##################################################

    def train(self, ea_itera):
        """
        Training and validating a network

        @param ea_itera: evolution iteration
        @return results_val: dict with validation results
        @return best_itera: best iteration when validating
        """

        logging.info("        Network_User: Train---->")

        logging.info("        Network_User:     Creating Dataloader---->")
        # Selecting the training sets, either train or train final (train  + Validation)
        if self.config["usage_modus"] == "train":
            harwindows_train = HARWindows(
                config = self.config,
                csv_file=self.config["dataset_root"] + "train.csv",
                root_dir=self.config["dataset_root"],
                type_dataset="train",
            )
        elif self.config["usage_modus"] == "train_final":
            harwindows_train = HARWindows(
                config=self.config,
                csv_file=self.config["dataset_root"] + "train_final.csv",
                root_dir=self.config["dataset_root"],
                type_dataset="train",
            )
        elif self.config["usage_modus"] == "fine_tuning":
            harwindows_train = HARWindows(
                config=self.config,
                csv_file=self.config["dataset_root"] + "train_final.csv", #"train.csv",
                root_dir=self.config["dataset_root"],
                type_dataset="train",
            )

        # Creating the dataloader
        dataLoader_train = DataLoader(harwindows_train, batch_size=self.config["batch_size_train"], shuffle=True)

        # Setting the network
        logging.info("        Network_User:    Train:    creating network")
        if self.config["network"] in ["cnn_imu"]:
            network_obj = Network(self.config)
            if self.config["usage_modus"] in ["train_final", "train"]:
                network_obj.init_weights()

            # Displaying size of tensors
            logging.info("        Network_User:    Train:    network layers")
            for l in list(network_obj.named_parameters()):
                logging.info("        Network_User:    Train:    {} : {}".format(l[0], l[1].detach().numpy().shape))
                if l[0] == "fc4.weight":
                    print(l[0])
                    print(l[1].detach().numpy()[0, :20])
                    print(l[1].detach().numpy()[0, -20:])

            if self.config["tensorboard_bool"]:
                self.exp.add_graph(
                    network_obj,
                    torch.randn(
                        [
                            self.config["batch_size_train"],
                            1,
                            self.config["sliding_window_length"],
                            self.config["NB_sensor_channels"],
                        ]
                    ),
                )
                self.exp.close()

            # IF finetuning, load the weights from a source dataset
            if self.config["usage_modus"] == "fine_tuning":
                logging.info("        Network_User:    Train:    Loading pre trained model")
                network_obj = self.load_weights(network_obj)
                #network_obj.load_state_dict(
                #    torch.load(self.config["folder_exp_base_fine_tuning"] + "network.pt")["state_dict"]
                #)

            # Displaying size of tensors
            logging.info("        Network_User:    Train:    \n")
            logging.info("        Network_User:    Train:    network layers")
            for l in list(network_obj.named_parameters()):
                logging.info("        Network_User:    Train:    {} : {}".format(l[0], l[1].detach().numpy().shape))
                if l[0] == "fc4.weight":
                    print(l[0])
                    print(l[1].detach().numpy()[0, :20])
                    print(l[1].detach().numpy()[0, -20:])
                if l[0] == "PROGLOVE.weight":
                    print(l[0])
                    print(l[1].detach().numpy()[0, :20])
                    print(l[1].detach().numpy()[0, -20:])

            # Displaying size of tensors
            # logging.info('        Network_User:    Train:    network layers')
            # for l in list(network_obj.named_parameters()):
            #    logging.info('        Network_User:    Train:    {} : {}'.format(l[0], l[1].detach().numpy().shape))

            logging.info("        Network_User:    Train:    setting device")
            network_obj.to(self.device)

        # Setting loss
        logging.info("        Network_User:    Train:    setting criterion optimizer Softmax")
        logging.info("        Network_User:    Train:    setting criterion optimizer Attribute")

        if True:
            if self.config["classification_target"] in ["BASE_ACTIVITIES"]:
                criterion_softmax = nn.CrossEntropyLoss(
                    weight=torch.tensor(
                        [0.216549515681407, 0.143808571875777, 0.303374801800306, 0.162305732776237, 0.173961377866272]
                    ).to(self.device, dtype=torch.float)
                )
            else:
                criterion_softmax = nn.CrossEntropyLoss()

        criterion_softmax = nn.CrossEntropyLoss()
        criterion_activity_softmax = nn.CrossEntropyLoss()
        criterion_windows_activity_softmax = nn.CrossEntropyLoss()
        criterion_statistics_activity_attribute = nn.BCELoss()
        criterion_attribute = nn.BCELoss()
        criterion_windows_attribute_softmax = nn.BCELoss()

        # Setting the freezing or not freezing from conv layers
        freezing = Freezing(self.config)
        if self.config["freeze_options"]:
            #network_obj = self.set_required_grad(network_obj)
            network_obj = freezing.set_required_grad_branches_on(network_obj)
        else:
            network_obj = freezing.set_required_default(network_obj)

        # Setting optimizer
        optimizer = optim.RMSprop(network_obj.parameters(), lr=self.config["lr"], alpha=0.95)

        # Setting scheduler
        if self.config["usage_modus"] in ["train_final", "fine_tuning"]:
            step_lr = self.config["epochs"]
        else:
            step_lr = self.config["epochs"] / 3
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=math.ceil(step_lr), gamma=0.5)

        # Initializing lists for plots and results
        losses_train = []
        accs_train = []
        f1w_train = []
        f1m_train = []

        losses_val = []
        accs_val = []
        f1w_val = []
        f1m_val = []

        loss_train_val = []
        accs_train_val = []
        f1w_train_val = []
        f1m_train_val = []

        best_loss_val = np.inf

        # initialising object for computing metrics
        metrics_obj = Metrics(self.config, self.device, self.attrs)

        itera = 0
        start_time_train = time.time()

        # zero the parameter gradients
        optimizer.zero_grad()

        best_itera = 0
        best_results_val = []
        itera = 0

        # loop for training
        # Validation is not carried out per epoch, but after certain number of iterations, specified
        # in configuration in main
        for e in range(self.config["epochs"]):
            start_time_train = time.time()
            logging.info("\n        Network_User:    Train:    Training epoch {}".format(e))
            start_time_batch = time.time()

            # loop per batch:
            for b, harwindow_batched in enumerate(dataLoader_train):
                start_time_batch = time.time()

                # Selecting batch
                train_batch_v = harwindow_batched["data"]
                #print("Labels: ", harwindow_batched["labels"].size())
                # overall process + acts + attrs
                train_batch_l_p_class = torch.argmax(harwindow_batched["labels"][:, :, range_scores['Process']], axis=2)
                train_batch_l_p_class = train_batch_l_p_class.mode(dim=1).values
                train_batch_l_p_class = train_batch_l_p_class.reshape(-1)

                train_batch_l_overall_a_class = torch.argmax(harwindow_batched["labels"][:, :, range_scores['Activity']], axis=2)
                train_batch_l_overall_a_class = train_batch_l_overall_a_class.mode(dim=1).values
                train_batch_l_overall_a_class = train_batch_l_overall_a_class.reshape(-1)

                train_batch_l_overall_attr_class = harwindow_batched["labels"][:, :, range_scores['attrs']]
                #print("attrs overall ", train_batch_l_overall_attr_class.size())
                train_batch_l_overall_attr_class = torch.sum(train_batch_l_overall_attr_class, axis=1) / 1000
                #print("attrs overall ", train_batch_l_overall_attr_class.size())

                train_batch_l_a_class = harwindow_batched["labels"][:, :, range_scores['Activity']] # input [B, 13, T]
                #print("Labels 0", train_batch_l_a_class.size())
                train_batch_l_a_class = train_batch_l_a_class.permute(0, 2, 1)  # input [B, 13, T]
                #print("Labels 1", train_batch_l_a_class.size())
                train_batch_l_a_class = train_batch_l_a_class.unfold(dimension=2, size=100, step=10)  # input [B, 13, W, T]
                #print("Labels 2", train_batch_l_a_class.size())
                train_batch_l_a_class = train_batch_l_a_class.permute(0, 2, 3, 1)  # input [B, W, T, 13]
                #print("Labels 3", train_batch_l_a_class.size())
                batch_windows_classes = train_batch_l_a_class.shape[0]
                windows_windows_classes = train_batch_l_a_class.shape[1]
                #print("batch", batch_windows_classes, "windows", windows_windows_classes)
                train_batch_l_a_class = train_batch_l_a_class.reshape(-1, train_batch_l_a_class.size()[2], train_batch_l_a_class.size()[3]) # input [B x W, T, 13]
                #print("Labels 4", train_batch_l_a_class.size())
                train_batch_l_a_class = torch.argmax(train_batch_l_a_class, axis=2) # input [B x W, T x 1]
                #print("Labels 4", train_batch_l_a_class.size())
                #statistics_classes = torch.bincount(train_batch_l_a_class[0], minlength=self.config["train_show"])
                #print("statistics side windows reshaped", statistics_classes.size())
                #print("Labels 5", train_batch_l_a_class.size())

                train_batch_l_attrs_class = harwindow_batched["labels"][:, :, range_scores['attrs']]
                #print("attrs: raw", train_batch_l_attrs_class.size())
                train_batch_l_attrs_class = train_batch_l_attrs_class.permute(0, 2, 1)  # input [B, 1, C, T]
                #print("attrs: time end", train_batch_l_attrs_class.size())
                train_batch_l_attrs_class = train_batch_l_attrs_class.unfold(dimension=2, size=100, step=10)  # input [B, 1, C, W, T]
                #print("attrs: undolding", train_batch_l_attrs_class.size())
                train_batch_l_attrs_class = train_batch_l_attrs_class.permute(0, 2, 3, 1)  # input [B, W, 1, T, C]
                #print("attrs: permute", train_batch_l_attrs_class.size())
                train_batch_l_attrs_class = train_batch_l_attrs_class.reshape(-1, train_batch_l_attrs_class.size()[2], train_batch_l_attrs_class.size()[3])
                #print("attrs: reshape", train_batch_l_attrs_class.size())
                #print("attrs: indices", train_batch_l_a_class.mode(dim=0).indices)
                #print("attrs: size indices", train_batch_l_a_class.mode(dim=0).indices.size())
                #train_batch_l_attr = train_batch_l_attrs_class[:, 50, :]
                train_batch_l_attr = torch.sum(train_batch_l_attrs_class, axis=1) / 100
                #print("attrs: final ", train_batch_l_attr.size())
                #print("attrs: final ", train_batch_l_attr[0])

                train_batch_l_a_class = train_batch_l_a_class.mode(dim=1).values # input [B x W, T x 1]
                #print("Labels 6", train_batch_l_a_class.size())
                train_batch_l_a_class = train_batch_l_a_class.reshape(-1) # input [B x W, T x 1]
                #print("Labels 7", train_batch_l_a_class.size())

                # print("side windows", train_batch_l_a_class.size())
                statistics_windows_classes = train_batch_l_a_class.reshape(batch_windows_classes, windows_windows_classes)
                statistics_classes = torch.zeros((batch_windows_classes, self.config["num_classes"]))
                #print("side windows reshaped", statistics_windows_classes.size())
                for b_statistic in range(batch_windows_classes):
                    statistics_classes[b_statistic] = torch.bincount(statistics_windows_classes[b_statistic],
                                                                     minlength=self.config["num_classes"]) / windows_windows_classes
                #print("statistics side windows reshaped", statistics_classes.size())
                #print("Statistics side windows labels", statistics_classes)

                sys.stdout.write(
                    "\rTraining: Epoch {}/{} Batch {}/{} Itera {} with {} Size {}".format(
                        e,
                        self.config["epochs"],
                        b,
                        len(dataLoader_train),
                        itera,
                        torch.bincount(train_batch_l_a_class, minlength=self.config["num_classes"]),
                        harwindow_batched["data"].size()
                        # torch.mean(harwindow_batched["data"][0, 0], axis=0),
                    )
                )
                sys.stdout.flush()
                #print(torch.bincount(train_batch_l_a_class, minlength=self.config["num_classes"]))
                # print(torch.mean(harwindow_batched["data"][0, 0], axis=0).size())

                # Setting the network to train mode
                network_obj.train(mode=True)

                # Adding gaussian noise
                noise_add = self.normal.sample((train_batch_v.size()))
                noise_add = noise_add.reshape(train_batch_v.size())
                noise_add = noise_add.to(self.device, dtype=torch.float)

                # Multi gaussian noise
                noise_mult = self.normal_mult.sample((train_batch_v.size()))
                noise_mult = noise_mult.reshape(train_batch_v.size())
                noise_mult = noise_mult.to(self.device, dtype=torch.float)

                # Sending to GPU
                train_batch_v = train_batch_v.to(self.device, dtype=torch.float)
                train_batch_v *= noise_mult
                train_batch_v += noise_add
                train_batch_l_p_class = train_batch_l_p_class.to(
                    self.device, dtype=torch.long
                )  # labels for crossentropy needs long type
                train_batch_l_overall_a_class = train_batch_l_overall_a_class.to(
                    self.device, dtype=torch.long
                )
                train_batch_l_a_class = train_batch_l_a_class.to(
                    self.device, dtype=torch.long
                )
                train_batch_l_attr = train_batch_l_attr.to(
                    self.device, dtype=torch.float
                )  # labels for binerycrossentropy needs float type
                train_batch_l_overall_attr_class = train_batch_l_overall_attr_class.to(
                    self.device, dtype=torch.float
                )  # labels for binerycrossentropy needs float type
                statistics_classes = statistics_classes.to(
                    self.device, dtype=torch.float
                )  # labels for binerycrossentropy needs float type

                #print(train_batch_l_attr.size())

                # forward + backward + optimize
                if self.config["frequency"] in ["75"]:
                    idx_frequency = np.arange(0, 100, 1)
                    train_batch_v = train_batch_v[:, :, idx_frequency, :]
                elif self.config["frequency"] in ["50"]:
                    idx_frequency = np.arange(0, 100, 2)
                    train_batch_v = train_batch_v[:, :, idx_frequency, :]
                elif self.config["frequency"] in ["25"]:
                    idx_frequency = np.arange(0, 100, 4)
                    train_batch_v = train_batch_v[:, :, idx_frequency, :]

                # NEtwork predictions
                #print("Device inpt", train_batch_v.get_device())
                overall_predictions, window_predictions, windows_attrs = network_obj(train_batch_v)
                #print(" windows_attrs", windows_attrs.size())

                class_predictions = overall_predictions[:, range_scores["Process"]]
                activity_predictions = overall_predictions[:, range_scores["Activity"]]
                attrs_overall_predictions = overall_predictions[:, range_scores["attrs"]]

                # print("side windows", train_batch_l_a_class.size())
                #print("side windows", window_predictions.size())
                statistics_window_predictions = window_predictions.reshape(batch_windows_classes, windows_windows_classes,
                                                                window_predictions.size()[1])
                #print("side windows reshaped", statistics_window_predictions.size())
                statistics_window_predictions = torch.argmax(statistics_window_predictions, axis=2)
                #print("side windows reshaped", statistics_window_predictions.size())
                statistics_prediction_classes = torch.zeros((batch_windows_classes, self.config["num_classes"]))
                #print("side windows reshaped", statistics_prediction_classes.size())
                for b_statistic in range(batch_windows_classes):
                    statistics_prediction_classes[b_statistic] = torch.bincount(statistics_window_predictions[b_statistic],
                                                                                minlength=self.config["num_classes"]) / windows_windows_classes
                #print("statistics side windows reshaped", statistics_prediction_classes.size())
                #print("Statistics side windows predictions", statistics_prediction_classes)

                #print("loss 1")
                statistics_prediction_classes = statistics_prediction_classes.to(
                    self.device, dtype=torch.float
                )  # labels for binerycrossentropy needs float type

                #process
                #print("loss 2")
                loss_softmax = criterion_softmax(class_predictions, train_batch_l_p_class) * (
                    1 / self.config["accumulation_steps"]
                )

                #activity global
                #print("loss 3")
                loss_activity_softmax = criterion_activity_softmax(activity_predictions, train_batch_l_overall_a_class) * (
                    1 / self.config["accumulation_steps"]
                )

                # activity windows
                #print("loss 4")
                loss_windows_softmax = criterion_windows_activity_softmax(window_predictions, train_batch_l_a_class) * (
                    1 / self.config["accumulation_steps"]
                )

                # attributes overall
                #print("loss 5")
                loss_attribute = criterion_attribute(attrs_overall_predictions, train_batch_l_overall_attr_class) * (
                    1 / self.config["accumulation_steps"]
                )

                # attributes windows
                #print("loss 6")
                loss_windows_attribute = criterion_windows_attribute_softmax(windows_attrs, train_batch_l_attr) * (
                    1 / self.config["accumulation_steps"]
                )

                # statistics windows
                #print("loss 7")
                #print(type(statistics_prediction_classes), type(statistics_classes))
                loss_statistics_classes = criterion_statistics_activity_attribute(statistics_prediction_classes, statistics_classes) * (
                    1 / self.config["accumulation_steps"]
                )

                if self.config["classification_target"] == 'Process':
                    loss = loss_softmax
                elif self.config["classification_target"] == 'Activity':
                    loss = loss_activity_softmax
                elif self.config["classification_target"] == 'Activity_windows':
                    loss = loss_windows_softmax
                elif self.config["classification_target"] == 'Attributes':
                    loss = loss_attribute
                elif self.config["classification_target"] == 'Attributes_windows':
                    loss = loss_windows_attribute
                elif self.config["classification_target"] == 'Statistics_activities':
                    loss = loss_statistics_classes
                else:
                    loss = 0.16 * loss_softmax + 0.16 * loss_activity_softmax + 0.2 * loss_windows_softmax + \
                           0.16 * loss_attribute + 0.16 * loss_windows_attribute + 0.16 * loss_statistics_classes
                #else:
                #    loss = loss_softmax
                loss.backward()

                if (itera + 1) % self.config["accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(network_obj.parameters(), max_norm=10.0)
                    optimizer.step()
                    # zero the parameter gradients
                    optimizer.zero_grad()

                loss_train = loss.item()

                elapsed_time_batch = time.time() - start_time_batch

                ################################## Validating ##################################################

                if (itera + 1) % self.config["valid_show"] == 0 or (itera + 1) == (
                    self.config["epochs"] * harwindow_batched["data"].shape[0]
                ):
                    logging.info("\n")
                    logging.info("        Network_User:        Validating")
                    start_time_val = time.time()

                    # Setting the network to eval mode
                    network_obj.eval()

                    # Metrics for training for keeping the same number of metrics for val and training
                    # Metrics return a dict with the metrics.
                    metrics_obj.metric(targets=train_batch_l_p_class,
                                       predictions=class_predictions,
                                       classification_target="Process")
                    metrics_obj.metric(targets=train_batch_l_overall_a_class,
                                       predictions=activity_predictions,
                                       classification_target="Activity")
                    metrics_obj.metric(targets=train_batch_l_a_class,
                                       predictions=window_predictions,
                                       classification_target="Activity_windows")

                    results_train = metrics_obj.return_results()

                    loss_train_val.append(loss_train)
                    accs_train_val.append(results_train[self.config["classification_target"]]["acc"])
                    f1w_train_val.append(results_train[self.config["classification_target"]]["f1_weighted"])
                    f1m_train_val.append(results_train[self.config["classification_target"]]["f1_mean"])

                    # Validation
                    # Calling the val() function with the current network and criterion
                    del train_batch_v, noise_add, noise_mult
                    # results_val, loss_val = self.validate(network_obj, [criterion_softmax, criterion_attribute])
                    results_val, loss_val = self.validate(network_obj)

                    elapsed_time_val = time.time() - start_time_val

                    # Appending the val metrics
                    losses_val.append(loss_val)
                    accs_val.append(results_val[self.config["classification_target"]]["acc"])
                    f1w_val.append(results_val[self.config["classification_target"]]["f1_weighted"])
                    f1m_val.append(results_val[self.config["classification_target"]]["f1_mean"])
                    if self.config["tensorboard_bool"]:
                        self.exp.add_scalar(
                            "Loss_TrainVal_inTrain_{}".format(ea_itera), scalar_value=loss_train, global_step=itera
                        )
                        self.exp.add_scalar(
                            "Acc_TrainVal_inTrain_{}".format(ea_itera),
                            scalar_value=results_train[self.config["classification_target"]]["acc"],
                            global_step=itera,
                        )
                        self.exp.add_scalar(
                            "F1w_TrainVal_inTrain_{}".format(ea_itera),
                            scalar_value=results_train[self.config["classification_target"]]["f1_weighted"],
                            global_step=itera,
                        )
                        self.exp.add_scalar(
                            "F1m_TrainVal_inTrain_{}".format(ea_itera),
                            scalar_value=results_train[self.config["classification_target"]]["f1_mean"],
                            global_step=itera,
                        )
                        self.exp.add_scalar(
                            "Loss_Val_inTrain_{}".format(ea_itera), scalar_value=loss_val, global_step=itera
                        )
                        self.exp.add_scalar(
                            "Acc_Val_inTrain_{}".format(ea_itera).format(),
                            scalar_value=results_val[self.config["classification_target"]]["acc"],
                            global_step=itera,
                        )
                        self.exp.add_scalar(
                            "F1w_Val_inTrain_{}".format(ea_itera),
                            scalar_value=results_val[self.config["classification_target"]]["f1_weighted"],
                            global_step=itera,
                        )
                        self.exp.add_scalar(
                            "F1m_Val_inTrain_{}".format(ea_itera),
                            scalar_value=results_val[self.config["classification_target"]]["f1_mean"],
                            global_step=itera,
                        )

                    # print statistics
                    logging.info("\n")
                    logging.info(
                        "        Network_User:        Validating:    "
                        "epoch {} batch {} itera {} elapsed time {}, best itera {}".format(
                            e, b, itera, elapsed_time_val, best_itera
                        )
                    )
                    logging.info(
                        "        Network_User:        Validating:    "
                        "acc {}, f1_weighted {}, f1_mean {}".format(
                            results_val[self.config["classification_target"]]["acc"],
                            results_val[self.config["classification_target"]]["f1_weighted"],
                            results_val[self.config["classification_target"]]["f1_mean"],
                        )
                    )
                    # Saving the network for the less iteration loss
                    logging.info(
                        "        Network_User:        Validating: Best loss {} New Loss {}".format(
                            best_loss_val, loss_val
                        )
                    )
                    if loss_val < best_loss_val:  # results_val['f1_weighted'] > best_acc_val:
                        if self.config["usage_modus"] == "fine_tuning":
                            classification_targets = torch.load(
                                self.config["folder_exp_base_fine_tuning"] + "network.pt"
                            )["network_config"]["classification_targets"]
                            classification_targets.append(self.config["classification_target"])
                        else:
                            classification_targets = [self.config["classification_target"]]
                        network_config = {
                            "NB_sensor_channels": self.config["NB_sensor_channels"],
                            "sliding_window_length": self.config["sliding_window_length"],
                            "filter_size": self.config["filter_size"],
                            "num_filters": self.config["num_filters"],
                            "reshape_input": self.config["reshape_input"],
                            "network": self.config["network"],
                            "output": self.config["output"],
                            "num_classes": self.config["num_classes"],
                            "num_attributes": self.config["num_attributes"],
                            "classification_targets": classification_targets,
                        }

                        logging.info("        Network_User:            Saving the network")

                        torch.save(
                            {
                                "state_dict": network_obj.state_dict(),
                                "network_config": network_config,
                                "att_rep": self.attr_representation,
                            },
                            self.config["folder_exp"] + "network.pt",
                        )

                        torch.save(network_obj.state_dict(), self.config["folder_exp"] + "state_dict.pt")

                        """
                        if self.config["usage_modus"] == "train_final":
                            torch.save({'state_dict': network_obj.state_dict(),
                                        'network_config': network_config,
                                        'att_rep': self.attr_representation},
                                       self.config['folder_exp_base_fine_tuning'] + 'network.pt')

                            torch.save(network_obj.state_dict(), self.config['folder_exp_base_fine_tuning'] + 'state_dict.pt')
                        print(self.config['folder_exp'])
                        """
                        best_loss_val = loss_val
                        # best_acc_val = results_val["classification"]['f1_weighted']
                        best_results_val = results_val
                        best_itera = itera
                    logging.info("\n")
                    logging.info("        Network_User:        Validating: After Storing Weights")

                # Computing metrics for current training batch
                if (itera) % self.config["train_show"] == 0:
                    # Metrics for training
                    metrics_obj.metric(targets=train_batch_l_p_class,
                                       predictions=class_predictions,
                                       classification_target="Process")
                    metrics_obj.metric(targets=train_batch_l_overall_a_class,
                                       predictions=activity_predictions,
                                       classification_target="Activity")
                    metrics_obj.metric(targets=train_batch_l_a_class,
                                       predictions=window_predictions,
                                       classification_target="Activity_windows")

                    results_train = metrics_obj.return_results()

                    #loss_softmax + loss_windows_softmax + loss_attribute
                    if self.config["tensorboard_bool"]:
                        self.exp.add_scalar(
                            "Loss_Train_inTrain_{}".format(ea_itera), scalar_value=loss_train, global_step=itera
                        )
                        self.exp.add_scalar(
                            "Loss_process_Train_inTrain_{}".format(ea_itera), scalar_value=loss_softmax, global_step=itera
                        )
                        self.exp.add_scalar(
                            "Loss_activity_Train_inTrain_{}".format(ea_itera), scalar_value=loss_activity_softmax, global_step=itera
                        )
                        self.exp.add_scalar(
                            "Loss_activity_windows_Train_inTrain_{}".format(ea_itera), scalar_value=loss_windows_softmax, global_step=itera
                        )
                        self.exp.add_scalar(
                            "Loss_attrs_Train_inTrain_{}".format(ea_itera), scalar_value=loss_attribute, global_step=itera
                        )
                        self.exp.add_scalar(
                            "Acc_Train_inTrain_{}".format(ea_itera),
                            scalar_value=results_train[self.config["classification_target"]]["acc"],
                            global_step=itera,
                        )
                        self.exp.add_scalar(
                            "F1w_Train_inTrain_{}".format(ea_itera),
                            scalar_value=results_train[self.config["classification_target"]]["f1_weighted"],
                            global_step=itera,
                        )
                        self.exp.add_scalar(
                            "F1m_Train_inTrain_{}".format(ea_itera),
                            scalar_value=results_train[self.config["classification_target"]]["f1_mean"],
                            global_step=itera,
                        )

                    activaciones = []
                    metrics_list = []
                    accs_train.append(results_train[self.config["classification_target"]]["acc"])
                    f1w_train.append(results_train[self.config["classification_target"]]["f1_weighted"])
                    f1m_train.append(results_train[self.config["classification_target"]]["f1_mean"])
                    losses_train.append(loss_train)

                    # print statistics
                    logging.info(
                        "        Network_User: Dataset {} network {} "
                        "pooling {} lr {} lr_optimizer {} Reshape {} "
                        "Freeze {} Aggregator {} "
                        "Magnetometer {}".format(
                            self.config["dataset"],
                            self.config["network"],
                            self.config["pooling"],
                            self.config["lr"],
                            optimizer.param_groups[0]["lr"],
                            self.config["reshape_input"],
                            self.config["freeze_options"],
                            self.config["aggregate"],
                            self.config["magnetometer"],
                        )
                    )
                    logging.info(
                        "        Network_User:    Train:    epoch {}/{} batch {}/{} itera {} "
                        "elapsed time {} best itera {}".format(
                            e, self.config["epochs"], b, len(dataLoader_train), itera, elapsed_time_batch, best_itera
                        )
                    )#loss_softmax + loss_windows_softmax + loss_attribute
                    logging.info("        Network_User:    Train:    loss {}, "
                                 "loss process {}, loss activities {}, "
                                 "loss attrs {}".format(loss, loss_softmax, loss_windows_softmax, loss_attribute))
                    logging.info(
                        "        Network_User:    Train:        Process acc {}, "
                        "f1_weighted {}, f1_mean {}".format(
                            results_train["Process"]["acc"],
                            results_train["Process"]["f1_weighted"],
                            results_train["Process"]["f1_mean"],
                        )
                    )
                    logging.info(
                        "        Network_User:    Train:        Activity acc {}, "
                        "f1_weighted {}, f1_mean {}".format(
                            results_train["Activity"]["acc"],
                            results_train["Activity"]["f1_weighted"],
                            results_train["Activity"]["f1_mean"],
                        )
                    )
                    logging.info(
                        "        Network_User:    Train:        Activitiy Windows acc {}, "
                        "f1_weighted {}, f1_mean {}".format(
                            results_train["Activity_windows"]["acc"],
                            results_train["Activity_windows"]["f1_weighted"],
                            results_train["Activity_windows"]["f1_mean"],
                        )
                    )
                    logging.info(
                        "        Network_User:    Train:    "
                        "Allocated {} GB Cached {} GB".format(
                            round(torch.cuda.memory_allocated(0) / 1024**3, 1),
                            round(torch.cuda.memory_cached(0) / 1024**3, 1),
                        )
                    )
                    logging.info(
                        "        Network_User:    Train:    "
                        "Predictions Process {}, Class {}, Process Target {}".format(network_obj.softmax(class_predictions)[0],
                                                                                     torch.argmax(class_predictions[0]),
                                                                                     train_batch_l_p_class[0]
                        )
                    )
                    logging.info(
                        "        Network_User:    Train:    "
                        "Predictions Activity_OverAll {}, Class {}, over all activities Target {}".format(network_obj.softmax(activity_predictions)[0],
                                                                                       torch.argmax(
                                                                                           activity_predictions[0]),
                                                                                       train_batch_l_overall_a_class[0]
                        )
                    )

                    #print("side windows", train_batch_l_a_class.size())
                    statistics_classes = train_batch_l_a_class.reshape(batch_windows_classes, windows_windows_classes)
                    #print("side windows reshaped", statistics_classes.size())
                    statistics_classes = torch.bincount(statistics_classes[0], minlength=self.config["num_classes"])
                    #print("statistics side windows reshaped", statistics_classes.size())

                    logging.info(
                        "        Network_User:    Train:    epoch {}/{} batch {}/{} itera {} "
                        "Statistics per 100 samples labels {}".format(
                            e, self.config["epochs"], b, len(dataLoader_train), itera, statistics_classes
                        )
                    )
                    logging.info(
                        "        Network_User:    Train:    "
                        "Predictions Activity_Windows {}, Class {}, Activity_Windows Target {}".format(network_obj.softmax(window_predictions)[0],
                                                                                       torch.argmax(
                                                                                           window_predictions[
                                                                                               0]),
                                                                                       train_batch_l_a_class[0]
                        )
                    )
                    #print("side windows", window_predictions.size())
                    window_predictions = window_predictions.reshape(batch_windows_classes, windows_windows_classes, window_predictions.size()[1])
                    #print("side windows reshaped", window_predictions.size())
                    window_predictions = torch.argmax(window_predictions[0], axis=1)
                    #print("side windows reshaped", window_predictions.size())
                    statistics_classes = torch.bincount(window_predictions, minlength=self.config["num_classes"])
                    #print("statistics side windows reshaped", statistics_classes.size())
                    logging.info(
                        "        Network_User:    Train:    epoch {}/{} batch {}/{} itera {} "
                        "statistics per 100 samples predictions {}".format(
                            e, self.config["epochs"], b, len(dataLoader_train), itera, statistics_classes
                        )
                    )
                    logging.info(
                        "        Network_User:    Train:    epoch {}/{} batch {}/{} itera {} "
                        "statistics per 100 samples predictions {}".format(
                            e, self.config["epochs"], b, len(dataLoader_train), itera, statistics_prediction_classes[0]
                        )
                    )
                    logging.info(
                        "        Network_User:    Train:    "
                        "Predictions {} \n  Attrs {}, Targets {}".format(network_obj.softmax(class_predictions)[0],
                                                                         attrs_overall_predictions[0], train_batch_l_p_class[0]
                        )
                    )
                    logging.info("\n\n--------------------------")



                itera += 1
            # Step of the the scheduler
            scheduler.step()

            if False:  # e < self.config['epochs'] - 5:
                with torch.no_grad():
                    for param in network_obj.parameters():
                        param.add_((torch.randn(param.size()) * 0.001).to(self.device, dtype=torch.float))

        elapsed_time_train = time.time() - start_time_train

        logging.info("\n")
        logging.info(
            "        Network_User:    Train:    epoch {} batch {} itera {} "
            "Total training time {}".format(e, b, itera, elapsed_time_train)
        )

        # Storing the acc, f1s and losses of training and validation for the current training run
        np.savetxt(self.config["folder_exp"] + "plots/acc_train_val.txt", accs_train_val, delimiter=",", fmt="%s")
        np.savetxt(self.config["folder_exp"] + "plots/f1m_train_val.txt", f1m_train_val, delimiter=",", fmt="%s")
        np.savetxt(self.config["folder_exp"] + "plots/f1w_train_val.txt", f1w_train_val, delimiter=",", fmt="%s")
        np.savetxt(self.config["folder_exp"] + "plots/loss_train_val.txt", loss_train_val, delimiter=",", fmt="%s")

        np.savetxt(self.config["folder_exp"] + "plots/acc_val.txt", accs_val, delimiter=",", fmt="%s")
        np.savetxt(self.config["folder_exp"] + "plots/f1m_val.txt", f1m_val, delimiter=",", fmt="%s")
        np.savetxt(self.config["folder_exp"] + "plots/f1w_val.txt", f1w_val, delimiter=",", fmt="%s")
        np.savetxt(self.config["folder_exp"] + "plots/loss_val.txt", losses_val, delimiter=",", fmt="%s")

        np.savetxt(self.config["folder_exp"] + "plots/acc_train.txt", accs_train, delimiter=",", fmt="%s")
        np.savetxt(self.config["folder_exp"] + "plots/f1m_train.txt", f1m_train, delimiter=",", fmt="%s")
        np.savetxt(self.config["folder_exp"] + "plots/f1w_train.txt", f1w_train, delimiter=",", fmt="%s")
        np.savetxt(self.config["folder_exp"] + "plots/loss_train.txt", losses_train, delimiter=",", fmt="%s")

        del losses_train, accs_train, f1w_train, f1m_train
        del losses_val, accs_val, f1w_val, f1m_val
        del loss_train_val, accs_train_val, f1w_train_val, f1m_train_val
        del metrics_list
        del network_obj

        torch.cuda.empty_cache()

        return best_results_val, best_itera

    ##################################################
    ################  Validate  ######################
    ##################################################

    def validate(self, network_obj):
        # def validate(self, network_obj, criterion):
        """
        Validating a network

        @param network_obj: network object
        @param criterion: torch criterion object
        @return results_val: dict with validation results
        @return loss: loss of the validation
        """

        # Setting validation set and dataloader
        harwindows_val = HARWindows(
            config=self.config,
            csv_file=self.config["dataset_root"] + "val.csv", root_dir=self.config["dataset_root"], type_dataset="val"
        )

        # Setting the network to eval mode
        dataLoader_val = DataLoader(harwindows_val, batch_size=self.config["batch_size_val"])

        # Setting the network to eval mode
        network_obj.eval()

        # Creating metric object
        metrics_obj = Metrics(self.config, self.device, self.attrs)
        loss_val = 0

        # Setting loss, only for being measured. Network wont be trained
        logging.info("        Network_User:    Val:    setting criterion optimizer Softmax")
        logging.info("        Network_User:    Val:    setting criterion optimizer Attribute")
        criterion = [nn.CrossEntropyLoss(),
                     nn.CrossEntropyLoss(),
                     nn.CrossEntropyLoss(),
                     nn.BCELoss(),
                     nn.BCELoss(),
                     nn.BCELoss()]


        # One doesnt need the gradients
        with torch.no_grad():
            for v, harwindow_batched_val in enumerate(dataLoader_val):
                # Selecting batch
                test_batch_v = harwindow_batched_val["data"]
                test_batch_l_p_class = torch.argmax(harwindow_batched_val["labels"][:, :, range_scores['Process']], axis=2)
                test_batch_l_p_class = test_batch_l_p_class.mode(dim=1).values
                test_batch_l_p_class = test_batch_l_p_class.reshape(-1)

                test_batch_l_overall_a_class = torch.argmax(harwindow_batched_val["labels"][:, :, range_scores['Activity']], axis=2)
                test_batch_l_overall_a_class = test_batch_l_overall_a_class.mode(dim=1).values
                test_batch_l_overall_a_class = test_batch_l_overall_a_class.reshape(-1)

                test_batch_l_overall_attr_class = harwindow_batched_val["labels"][:, :, range_scores['attrs']]
                test_batch_l_overall_attr_class = torch.sum(test_batch_l_overall_attr_class, axis=1) / 1000

                test_batch_l_a_class = harwindow_batched_val["labels"][:, :, range_scores['Activity']]
                test_batch_l_a_class = test_batch_l_a_class.permute(0, 2, 1)  # input [B, 1, C, T]
                test_batch_l_a_class = test_batch_l_a_class.unfold(dimension=2, size=100, step=10)  # input [B, 1, C, W, T]
                test_batch_l_a_class = test_batch_l_a_class.permute(0, 2, 3, 1)  # input [B, W, 1, T, C]
                batch_windows_classes = test_batch_l_a_class.shape[0]
                windows_windows_classes = test_batch_l_a_class.shape[1]
                test_batch_l_a_class = test_batch_l_a_class.reshape(-1, test_batch_l_a_class.size()[2], test_batch_l_a_class.size()[3])
                test_batch_l_a_class = torch.argmax(test_batch_l_a_class, axis=2)
                test_batch_l_a_class = test_batch_l_a_class.mode(dim=1).values
                test_batch_l_a_class = test_batch_l_a_class.reshape(-1)

                test_batch_l_attrs_class = harwindow_batched_val["labels"][:, :, range_scores['attrs']]
                test_batch_l_attrs_class = test_batch_l_attrs_class.permute(0, 2, 1)  # input [B, 1, C, T]
                test_batch_l_attrs_class = test_batch_l_attrs_class.unfold(dimension=2, size=100, step=10)  # input [B, 1, C, W, T]
                test_batch_l_attrs_class = test_batch_l_attrs_class.permute(0, 2, 3, 1)  # input [B, W, 1, T, C]
                test_batch_l_attrs_class = test_batch_l_attrs_class.reshape(-1, test_batch_l_attrs_class.size()[2], test_batch_l_attrs_class.size()[3])
                test_batch_l_attr = torch.sum(test_batch_l_attrs_class, axis=1) / 100

                statistics_windows_classes = test_batch_l_a_class.reshape(batch_windows_classes, windows_windows_classes)
                statistics_classes = torch.zeros((batch_windows_classes, self.config["num_classes"]))
                #print("side windows reshaped", statistics_windows_classes.size())
                for b_statistic in range(batch_windows_classes):
                    statistics_classes[b_statistic] = torch.bincount(statistics_windows_classes[b_statistic],
                                                                     minlength=self.config["num_classes"]) / windows_windows_classes
                #print("statistics side windows reshaped", statistics_classes.size())
                #print("Statistics side windows labels", statistics_classes)

                # Creating torch tensors
                test_batch_v = test_batch_v.to(self.device, dtype=torch.float)
                test_batch_l_p_class = test_batch_l_p_class.to(self.device, dtype=torch.long)
                test_batch_l_overall_a_class = test_batch_l_overall_a_class.to(
                    self.device, dtype=torch.long
                )
                test_batch_l_overall_attr_class = test_batch_l_overall_attr_class.to(
                    self.device, dtype=torch.float
                )  # labels for binerycrossentropy needs float type
                test_batch_l_a_class = test_batch_l_a_class.to(self.device, dtype=torch.long)
                test_batch_l_attr = test_batch_l_attr.to(
                    self.device, dtype=torch.float
                )  # labels for binerycrossentropy needs float type

                if self.config["frequency"] in ["75"]:
                    idx_frequency = np.arange(0, 100, 1)
                    test_batch_v = test_batch_v[:, :, idx_frequency, :]
                elif self.config["frequency"] in ["50"]:
                    idx_frequency = np.arange(0, 100, 2)
                    test_batch_v = test_batch_v[:, :, idx_frequency, :]
                elif self.config["frequency"] in ["25"]:
                    idx_frequency = np.arange(0, 100, 4)
                    test_batch_v = test_batch_v[:, :, idx_frequency, :]


                # Forward Pass
                overall_predictions, window_predictions, windows_attrs = network_obj(test_batch_v)
                class_predictions = overall_predictions[:, range_scores["Process"]]
                activity_predictions = overall_predictions[:, range_scores["Activity"]]
                attrs_overall_predictions = overall_predictions[:, range_scores["attrs"]]

                statistics_window_predictions = window_predictions.reshape(batch_windows_classes, windows_windows_classes,
                                                                window_predictions.size()[1])
                statistics_window_predictions = torch.argmax(statistics_window_predictions, axis=2)
                statistics_prediction_classes = torch.zeros((batch_windows_classes, self.config["num_classes"]))
                for b_statistic in range(batch_windows_classes):
                    statistics_prediction_classes[b_statistic] = torch.bincount(statistics_window_predictions[b_statistic],
                                                                                minlength=self.config["num_classes"]) / windows_windows_classes

                # loss process
                loss_softmax = criterion[0](class_predictions, test_batch_l_p_class)

                #activity global
                #print("loss 3")
                loss_activity_softmax = criterion[1](activity_predictions, test_batch_l_overall_a_class)

                # activity windows
                #print("loss 4")
                loss_windows_softmax = criterion[0](window_predictions, test_batch_l_a_class)

                # attributes overall
                #print("loss 5")
                loss_attribute = criterion[3](attrs_overall_predictions, test_batch_l_overall_attr_class)

                # attributes windows
                #print("loss 6")
                loss_windows_attribute = criterion[4](windows_attrs, test_batch_l_attr)

                # statistics windows
                loss_statistics_classes = criterion[5](statistics_prediction_classes, statistics_classes)

                if self.config["classification_target"] == 'Process':
                    loss = loss_softmax
                elif self.config["classification_target"] == 'Activity':
                    loss = loss_activity_softmax
                elif self.config["classification_target"] == 'Activity_windows':
                    loss = loss_windows_softmax
                elif self.config["classification_target"] == 'Attributes':
                    loss = loss_attribute
                elif self.config["classification_target"] == 'Attributes_windows':
                    loss = loss_windows_attribute
                elif self.config["classification_target"] == 'Statistics_activities':
                    loss = loss_statistics_classes
                else:
                    loss = 0.16 * loss_softmax + 0.16 * loss_activity_softmax + 0.2 * loss_windows_softmax + \
                           0.16 * loss_attribute + 0.16 * loss_windows_attribute + 0.16 * loss_statistics_classes

                loss_val += loss.item()

                # Concatenating all of the batches for computing the metrics
                # As creating an empty tensor and sending to device and then concatenating isnt working
                if v == 0:
                    if self.config["classification_target"] == 'Process':
                        predictions_val = class_predictions
                        test_labels = test_batch_l_p_class
                    elif self.config["classification_target"] == 'Activity':
                        predictions_val = activity_predictions
                        test_labels = test_batch_l_overall_a_class
                    elif self.config["classification_target"] == 'Activity_windows':
                        predictions_val = window_predictions
                        test_labels = test_batch_l_a_class
                    elif self.config["classification_target"] == 'Attributes':
                        predictions_val = window_predictions
                        test_labels = test_batch_l_a_class
                    elif self.config["classification_target"] == 'Attributes_windows':
                        predictions_val = window_predictions
                        test_labels = test_batch_l_a_class
                    elif self.config["classification_target"] == 'Statistics_activities':
                        predictions_val = window_predictions
                        test_labels = test_batch_l_a_class
                    else:
                        predictions_val = window_predictions
                        test_labels = test_batch_l_a_class

                else:
                    if self.config["classification_target"] == 'Process':
                        predictions_val = torch.cat((predictions_val, class_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_p_class), dim=0)
                    elif self.config["classification_target"] == 'Activity':
                        predictions_val = torch.cat((predictions_val, activity_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_overall_a_class), dim=0)
                    elif self.config["classification_target"] == 'Activity_windows':
                        predictions_val = torch.cat((predictions_val, window_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_a_class), dim=0)
                    elif self.config["classification_target"] == 'Attributes':
                        predictions_val = torch.cat((predictions_val, window_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_a_class), dim=0)
                    elif self.config["classification_target"] == 'Attributes_windows':
                        predictions_val = torch.cat((predictions_val, window_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_a_class), dim=0)
                    elif self.config["classification_target"] == 'Statistics_activities':
                        predictions_val = torch.cat((predictions_val, window_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_a_class), dim=0)
                    else:
                        predictions_val = torch.cat((predictions_val, window_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_a_class), dim=0)

                sys.stdout.write("\rValidating: Batch  {}/{}".format(v, len(dataLoader_val)))
                sys.stdout.flush()

        print("\n")
        print('test_labels: ', test_labels.size(), 'predictions_val: ', predictions_val.size())
        metrics_obj.metric(test_labels, predictions_val,
                           classification_target=self.config["classification_target"])

        results_val = metrics_obj.return_results()

        del test_batch_v, test_batch_l_a_class
        del class_predictions, predictions_val
        del test_labels, activity_predictions
        del window_predictions, test_batch_l_overall_a_class
        del test_batch_l_p_class, test_batch_l_overall_attr_class
        del test_batch_l_attr, statistics_classes
        del statistics_prediction_classes, windows_attrs

        torch.cuda.empty_cache()

        return results_val, loss_val / v

    ##################################################
    ###################  test  ######################
    ##################################################

    def test(self, ea_itera, ifVal=False):
        """
        Testing a network

        @param ea_itera: evolution iteration
        @return results_test: dict with testing results
        @return confusion matrix: confusion matrix of testing results
        """
        logging.info("        Network_User:    Test ---->")

        # Setting the testing set
        if ifVal:
            logging.info("        Network_User:     Creating Dataloader---->")
            harwindows_test = HARWindows(
                config=self.config,
                csv_file=self.config["dataset_root"] + "val.csv",
                root_dir=self.config["dataset_root"],
                type_dataset="val",
            )
        else:
            logging.info("        Network_User:     Creating Dataloader---->")
            harwindows_test = HARWindows(
                config=self.config,
                csv_file=self.config["dataset_root"] + "test_BBr_MS1_P1-1.csv",
                root_dir=self.config["dataset_root"],
                type_dataset="test",
            )

        dataLoader_test = DataLoader(harwindows_test, batch_size=1, shuffle=False)

        # Creating a network and loading the weights for testing
        # network is loaded from saved file in the folder of experiment
        logging.info("        Network_User:    Test:    creating network")
        if self.config["network"] in ["cnn", "cnn_imu", "cnn_tpp", "cnn_imu_tpp"]:
            network_obj = Network(self.config)
            self.exp.add_graph(
                network_obj,
                torch.randn(
                    [
                        self.config["batch_size_train"],
                        1,
                        self.config["sliding_window_length"],
                        self.config["NB_sensor_channels"],
                    ]
                ),
            )
            self.exp.close()

            # Loading the model
            # network_obj.load_state_dict(torch.load(self.config['folder_exp'] + 'network.pt', map_location=torch.device('cpu'))['state_dict'])
            network_obj.load_state_dict(torch.load(self.config["folder_exp"] + "network.pt")["state_dict"])
            network_obj.eval()
            logging.info(
                "        Network_User:    Test:    loaded network from \n{}".format(
                    (self.config["folder_exp"] + "network.pt")
                )
            )

            print(torch.load(self.config["folder_exp"] + "network.pt")["network_config"])

            # Displaying size of tensors
            logging.info("        Network_User:    Test:    network layers")
            for l in list(network_obj.named_parameters()):
                logging.info("        Network_User:    Test:    {} : {}".format(l[0], l[1].detach().numpy().shape))
                if l[0] == "fc4.weight":
                    print(l[0])
                    print(l[1].detach().numpy()[0, :20])
                    print(l[1].detach().numpy()[0, -20:])
                if l[0] == "PROGLOVE.weight":
                    print(l[0])
                    print(l[1].detach().numpy()[0, :20])
                    print(l[1].detach().numpy()[0, -20:])

            torch.save(network_obj.state_dict(), self.config["folder_exp"] + "state_dict.pt")

            logging.info("        Network_User:    Test:    setting device")
            network_obj.to(self.device)

        # Setting loss, only for being measured. Network wont be trained
        logging.info("        Network_User:    Train:    setting criterion optimizer Softmax")
        logging.info("        Network_User:    Train:    setting criterion optimizer Attribute")
        criterion_softmax = nn.CrossEntropyLoss()
        criterion_activity_softmax = nn.CrossEntropyLoss()
        criterion_windows_activity_softmax = nn.CrossEntropyLoss()
        criterion_statistics_activity_attribute = nn.BCELoss()
        criterion_attribute = nn.BCELoss()
        criterion_windows_attribute_softmax = nn.BCELoss()

        loss_test = 0

        # Creating metric object
        metrics_obj = Metrics(self.config, self.device, self.attrs, True)

        logging.info("        Network_User:    Testing")
        start_time_test = time.time()
        # loop for testing
        with torch.no_grad():
            for v, harwindow_batched_test in enumerate(dataLoader_test):
                # Selecting batch
                test_batch_v = harwindow_batched_test["data"]
                test_batch_l_p_class = torch.argmax(harwindow_batched_test["labels"][:, :, range_scores['Process']], axis=2)
                test_batch_l_p_class = test_batch_l_p_class.mode(dim=1).values
                test_batch_l_p_class = test_batch_l_p_class.reshape(-1)

                test_batch_l_overall_a_class = torch.argmax(harwindow_batched_test["labels"][:, :, range_scores['Activity']], axis=2)
                test_batch_l_overall_a_class = test_batch_l_overall_a_class.mode(dim=1).values
                test_batch_l_overall_a_class = test_batch_l_overall_a_class.reshape(-1)

                test_batch_l_overall_attr_class = harwindow_batched_test["labels"][:, :, range_scores['attrs']]
                test_batch_l_overall_attr_class = torch.sum(test_batch_l_overall_attr_class, axis=1) / 1000

                test_batch_l_a_class = harwindow_batched_test["labels"][:, :, range_scores['Activity']]
                test_batch_l_a_class = test_batch_l_a_class.permute(0, 2, 1)  # input [B, 1, C, T]
                test_batch_l_a_class = test_batch_l_a_class.unfold(dimension=2, size=100, step=10)  # input [B, 1, C, W, T]
                test_batch_l_a_class = test_batch_l_a_class.permute(0, 2, 3, 1)  # input [B, W, 1, T, C]
                batch_windows_classes = test_batch_l_a_class.shape[0]
                windows_windows_classes = test_batch_l_a_class.shape[1]
                test_batch_l_a_class = test_batch_l_a_class.reshape(-1, test_batch_l_a_class.size()[2], test_batch_l_a_class.size()[3])
                test_batch_l_a_class = torch.argmax(test_batch_l_a_class, axis=2)
                test_batch_l_a_class = test_batch_l_a_class.mode(dim=1).values
                test_batch_l_a_class = test_batch_l_a_class.reshape(-1)

                test_batch_l_attrs_class = harwindow_batched_test["labels"][:, :, range_scores['attrs']]
                test_batch_l_attrs_class = test_batch_l_attrs_class.permute(0, 2, 1)  # input [B, 1, C, T]
                test_batch_l_attrs_class = test_batch_l_attrs_class.unfold(dimension=2, size=100, step=10)  # input [B, 1, C, W, T]
                test_batch_l_attrs_class = test_batch_l_attrs_class.permute(0, 2, 3, 1)  # input [B, W, 1, T, C]
                test_batch_l_attrs_class = test_batch_l_attrs_class.reshape(-1, test_batch_l_attrs_class.size()[2], test_batch_l_attrs_class.size()[3])
                test_batch_l_attr = torch.sum(test_batch_l_attrs_class, axis=1) / 100

                statistics_windows_classes = test_batch_l_a_class.reshape(batch_windows_classes, windows_windows_classes)
                statistics_classes = torch.zeros((batch_windows_classes, self.config["num_classes"]))
                #print("side windows reshaped", statistics_windows_classes.size())
                for b_statistic in range(batch_windows_classes):
                    statistics_classes[b_statistic] = torch.bincount(statistics_windows_classes[b_statistic],
                                                                     minlength=self.config["num_classes"]) / windows_windows_classes

                # Creating torch tensors
                test_batch_v = test_batch_v.to(self.device, dtype=torch.float)
                test_batch_l_p_class = test_batch_l_p_class.to(self.device, dtype=torch.long)
                test_batch_l_overall_a_class = test_batch_l_overall_a_class.to(
                    self.device, dtype=torch.long
                )
                test_batch_l_overall_attr_class = test_batch_l_overall_attr_class.to(
                    self.device, dtype=torch.float
                )  # labels for binerycrossentropy needs float type
                test_batch_l_a_class = test_batch_l_a_class.to(self.device, dtype=torch.long)
                test_batch_l_attr = test_batch_l_attr.to(
                    self.device, dtype=torch.float
                )  # labels for binerycrossentropy needs float type


                if self.config["frequency"] in ["75"]:
                    idx_frequency = np.arange(0, 100, 1)
                    test_batch_v = test_batch_v[:, :, idx_frequency, :]
                elif self.config["frequency"] in ["50"]:
                    idx_frequency = np.arange(0, 100, 2)
                    test_batch_v = test_batch_v[:, :, idx_frequency, :]
                elif self.config["frequency"] in ["25"]:
                    idx_frequency = np.arange(0, 100, 4)
                    test_batch_v = test_batch_v[:, :, idx_frequency, :]
                # forward
                overall_predictions, window_predictions, windows_attrs = network_obj(test_batch_v)
                class_predictions = overall_predictions[:, range_scores["Process"]]
                activity_predictions = overall_predictions[:, range_scores["Activity"]]
                attrs_overall_predictions = overall_predictions[:, range_scores["attrs"]]

                statistics_window_predictions = window_predictions.reshape(batch_windows_classes, windows_windows_classes,
                                                                window_predictions.size()[1])
                statistics_window_predictions = torch.argmax(statistics_window_predictions, axis=2)
                statistics_prediction_classes = torch.zeros((batch_windows_classes, self.config["num_classes"]))
                for b_statistic in range(batch_windows_classes):
                    statistics_prediction_classes[b_statistic] = torch.bincount(statistics_window_predictions[b_statistic],
                                                                                minlength=self.config["num_classes"]) / windows_windows_classes


                # loss process
                loss_softmax = criterion_softmax(class_predictions, test_batch_l_p_class)

                #activity global
                #print("loss 3")
                loss_activity_softmax = criterion_activity_softmax(activity_predictions, test_batch_l_overall_a_class)

                # activity windows
                #print("loss 4")
                loss_windows_softmax = criterion_windows_activity_softmax(window_predictions, test_batch_l_a_class)

                # attributes overall
                #print("loss 5")
                loss_attribute = criterion_attribute(attrs_overall_predictions, test_batch_l_overall_attr_class)

                # attributes windows
                #print("loss 6")
                loss_windows_attribute = criterion_windows_attribute_softmax(windows_attrs, test_batch_l_attr)

                # statistics windows
                loss_statistics_classes = criterion_statistics_activity_attribute(statistics_prediction_classes, statistics_classes)

                if self.config["classification_target"] == 'Process':
                    loss = loss_softmax
                elif self.config["classification_target"] == 'Activity':
                    loss = loss_activity_softmax
                elif self.config["classification_target"] == 'Activity_windows':
                    loss = loss_windows_softmax
                elif self.config["classification_target"] == 'Attributes':
                    loss = loss_attribute
                elif self.config["classification_target"] == 'Attributes_windows':
                    loss = loss_windows_attribute
                elif self.config["classification_target"] == 'Statistics_activities':
                    loss = loss_statistics_classes
                else:
                    loss = 0.16 * loss_softmax + 0.16 * loss_activity_softmax + 0.2 * loss_windows_softmax + \
                           0.16 * loss_attribute + 0.16 * loss_windows_attribute + 0.16 * loss_statistics_classes
                loss_test = loss_test + loss.item()

                #print("---------------\n", class_predictions[0, :8])

                # Concatenating all of the batches for computing the metrics for the entire testing set
                # and not only for a batch
                # As creating an empty tensor and sending to device and then concatenating isnt working
                if v == 0:
                    if self.config["classification_target"] == 'Process':
                        predictions_test = class_predictions
                        test_labels = test_batch_l_p_class
                    elif self.config["classification_target"] == 'Activity':
                        predictions_test = activity_predictions
                        test_labels = test_batch_l_overall_a_class
                    elif self.config["classification_target"] == 'Activity_windows':
                        predictions_test = window_predictions
                        test_labels = test_batch_l_a_class
                    elif self.config["classification_target"] == 'Attributes':
                        predictions_test = window_predictions
                        test_labels = test_batch_l_a_class
                    elif self.config["classification_target"] == 'Attributes_windows':
                        predictions_test = window_predictions
                        test_labels = test_batch_l_a_class
                    elif self.config["classification_target"] == 'Statistics_activities':
                        predictions_test = window_predictions
                        test_labels = test_batch_l_a_class
                    else:
                        predictions_test = window_predictions
                        test_labels = test_batch_l_a_class

                    test_file_labels = harwindow_batched_test["label_file"]
                    test_file_labels = test_file_labels.reshape(-1)

                else:

                    if self.config["classification_target"] == 'Process':
                        predictions_test = torch.cat((predictions_test, class_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_p_class), dim=0)
                    elif self.config["classification_target"] == 'Activity':
                        predictions_test = torch.cat((predictions_test, activity_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_overall_a_class), dim=0)
                    elif self.config["classification_target"] == 'Activity_windows':
                        predictions_test = torch.cat((predictions_test, window_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_a_class), dim=0)
                    elif self.config["classification_target"] == 'Attributes':
                        predictions_test = torch.cat((predictions_test, window_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_a_class), dim=0)
                    elif self.config["classification_target"] == 'Attributes_windows':
                        predictions_test = torch.cat((predictions_test, window_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_a_class), dim=0)
                    elif self.config["classification_target"] == 'Statistics_activities':
                        predictions_test = torch.cat((predictions_test, window_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_a_class), dim=0)
                    else:
                        predictions_test = torch.cat((predictions_test, window_predictions), dim=0)
                        #predictions_test_attrs = torch.cat((predictions_test_attrs, attrs_overall_predictions), dim=0)
                        test_labels = torch.cat((test_labels, test_batch_l_a_class), dim=0)
                        #test_labels_attributes = torch.cat((test_labels_attributes, test_batch_l_attribute[:, 1:]), dim=0)

                    test_file_labels_batch = harwindow_batched_test["label_file"]
                    test_file_labels_batch = test_file_labels_batch.reshape(-1)
                    test_file_labels = torch.cat((test_file_labels, test_file_labels_batch), dim=0)
                    # test_labels = torch.cat((test_labels, test_labels_batch), dim=0)
                    #test_labels_window = torch.cat((test_labels_window, test_labels_window_batch), dim=0)

                sys.stdout.write("\rTesting: Batch  {}/{}".format(v, len(dataLoader_test)))
                sys.stdout.flush()

        elapsed_time_test = time.time() - start_time_test

        # if self.config["fully_convolutional"] == "FCN":
        #    results_test_segment = metrics_obj.metric(test_labels, predictions_test, mode="segmentation")
        # else:
        # Computing metrics
        test_labels = test_labels.to(self.device, dtype=torch.float)
        logging.info("            Testing:    type targets vector: {}".format(test_labels.type()))
        # acc_test, f1_weighted_test, f1_mean_test, predictions_labels = metrics_obj.metric(test_labels, predictions_test)
        #print("test_labels_attributes: ", test_labels_attributes.size())
        #print("predictions_test_attrs: ", predictions_test_attrs.size())
        #results_test_attrs = metrics_obj.metric(test_labels_attributes, predictions_test_attrs, mode="classification",
        #                                        attrs_prediction_on=False)
        #results_test_attrs = results_test_attrs["classification"]["predicted_classes"]
        #print("Results : ", results_test_attrs.size())
        #print("test_labels : ", test_labels.size())
        metrics_obj.metric(test_labels, predictions_test, classification_target=self.config["classification_target"])
        results_test = metrics_obj.return_results()

        # print statistics
        logging.info(
            "        Network_User:        Testing:    elapsed time {} acc {}, f1_weighted {}, f1_mean {}".format(
                elapsed_time_test,
                results_test["classification"]["acc"],
                results_test["classification"]["f1_weighted"],
                results_test["classification"]["f1_mean"],
            )
        )

        # Storing predicitons attributes
        predictions_test = predictions_test.to("cpu", torch.double).numpy()
        np.savetxt(
            self.config["folder_exp"] + "plots/attr_predictions.txt",
            np.round(predictions_test, decimals=4),
            delimiter=",",
            fmt="%1.4f",
        )

        predictions_labels = results_test["classification"]["predicted_classes"].to("cpu", torch.double).numpy()
        test_labels = test_labels.to("cpu", torch.double).numpy()

        results_test_segment = results_test

        # Computing confusion matrix
        confusion_matrix = np.zeros((self.config["num_classes"], self.config["num_classes"]))

        for cl in range(self.config["num_classes"]):
            pos_tg = test_labels == cl
            pos_pred = predictions_labels[pos_tg]
            bincount = np.bincount(pos_pred.astype(int), minlength=self.config["num_classes"])
            confusion_matrix[cl, :] = bincount

        logging.info(
            "        Network_User:        Testing:    Confusion matrix \n{}\n".format(confusion_matrix.astype(int))
        )

        percentage_pred = []
        for cl in range(self.config["num_classes"]):
            pos_trg = np.reshape(test_labels, newshape=test_labels.shape[0]) == cl
            percentage_pred.append(confusion_matrix[cl, cl] / float(np.sum(pos_trg)))
            print("Class: ", cl, "Samples: ", float(np.sum(pos_trg)))
        percentage_pred = np.array(percentage_pred)

        logging.info("        Network_User:        Testing:    percentage Pred \n{}\n".format(percentage_pred))

        del test_batch_v#, test_batch_l_clas, test_batch_l_attribute
        del class_predictions, predictions_test
        del test_labels, predictions_labels
        del network_obj

        torch.cuda.empty_cache()

        return results_test_segment, confusion_matrix.astype(int)

    ##################################################
    ###################  Pseudo annotate  ######################
    ##################################################

    def pseudo_annotate(self):
        """
        Testing a network

        @param ea_itera: evolution iteration
        @return results_test: dict with testing results
        @return confusion matrix: confusion matrix of testing results
        """
        logging.info("        Network_User:    Pseudo annotating ---->")

        # Setting the testing set
        logging.info("        Network_User:     Creating Dataloader---->")
        harwindows_test = HARWindows(
            config=self.config,
            csv_file=self.config["dataset_root"] + "test_BBr_MS1_P1-1.csv",
            root_dir=self.config["dataset_root"],
            type_dataset="test",
        )

        dataLoader_test = DataLoader(harwindows_test, batch_size=1, shuffle=False)

        # Creating a network and loading the weights for testing
        # network is loaded from saved file in the folder of experiment
        logging.info("        Network_User:    Pseudo labeling:    creating network")
        if self.config["network"] in ["cnn", "cnn_imu", "cnn_tpp", "cnn_imu_tpp"]:
            network_obj = Network(self.config)

            # Loading the model
            keys_network = torch.load(self.config["folder_exp"] + "network.pt")["state_dict"].keys()
            for kk in keys_network:
                print(kk, torch.load(self.config["folder_exp"] + "network.pt")["state_dict"][kk].size())
            print(torch.load(self.config["folder_exp"] + "network.pt")["network_config"])
            for kk in network_obj.state_dict().keys():
                print(kk, network_obj.state_dict()[kk].size())
            network_obj.load_state_dict(torch.load(self.config["folder_exp"] + "network.pt")["state_dict"])
            network_obj.eval()

            print(torch.load(self.config["folder_exp"] + "network.pt")["network_config"])

            logging.info("        Network_User:    Pseudo labeling:    setting device")
            network_obj.to(self.device)

        # Setting loss, only for being measured. Network wont be trained
        if self.config["output"] == "softmax":
            logging.info("        Network_User:    Pseudo labeling:    setting criterion optimizer Softmax")
            criterion = nn.CrossEntropyLoss()
        elif self.config["output"] == "attribute":
            logging.info("        Network_User:    Pseudo labeling:    setting criterion optimizer Attribute")
            criterion = nn.BCELoss()

        loss_test = 0

        # Creating metric object
        metrics_obj = Metrics(self.config, self.device, self.attrs)

        logging.info("        Network_User:    Pseudo labeling {}".format(len(dataLoader_test)))
        start_time_test = time.time()
        # loop for testing
        with torch.no_grad():
            for v, harwindow_batched_test in enumerate(dataLoader_test):
                # Selecting batch
                print(v)
                test_batch_v = harwindow_batched_test["data"]
                test_batch_l_p_class = torch.argmax(harwindow_batched_test["labels"][:, :, range_scores['Process']], axis=2)
                test_batch_l_p_class = test_batch_l_p_class.mode(dim=1).values
                test_batch_l_p_class = test_batch_l_p_class.reshape(-1)
                print('test_batch_l_p_class')
                print(test_batch_l_p_class.size())
                print(test_batch_l_p_class)

                test_batch_l_a_class = harwindow_batched_test["labels"][:, :, range_scores['Activity']]
                test_batch_l_a_class = test_batch_l_a_class.permute(0, 2, 1)  # input [B, 1, C, T]
                test_batch_l_a_class = test_batch_l_a_class.unfold(dimension=2, size=100, step=10)  # input [B, 1, C, W, T]
                test_batch_l_a_class = test_batch_l_a_class.permute(0, 2, 3, 1)  # input [B, W, 1, T, C]
                b_windows_sample = test_batch_l_a_class.size()[0]
                n_windows_sample = test_batch_l_a_class.size()[1]
                test_batch_l_a_class = test_batch_l_a_class.reshape(-1, test_batch_l_a_class.size()[2], test_batch_l_a_class.size()[3])
                test_batch_l_a_class = torch.argmax(test_batch_l_a_class, axis=2)

                test_batch_l_a_class = test_batch_l_a_class.mode(dim=1).values
                test_batch_l_a_class = test_batch_l_a_class.reshape(-1)

                test_batch_l_attribute = harwindow_batched_test["labels"]

                # Sending to GPU
                test_batch_v = test_batch_v.to(self.device, dtype=torch.float)
                test_batch_l_p_class = test_batch_l_p_class.to(self.device, dtype=torch.long)
                test_batch_l_attribute = test_batch_l_attribute.to(
                    self.device, dtype=torch.float
                )  # labels for binerycrossentropy needs float type
                test_batch_l_a_class = test_batch_l_a_class.to(
                    self.device, dtype=torch.long
                )

                pseudo_annotated = harwindow_batched_test
                # logging.info("        Network_User:        Pseudo Annotating:    File\n{}\n".format(
                #    pseudo_annotated["file_name"]))
                # print(len(pseudo_annotated["file_name"]))

                # forward
                class_predictions, window_predictions, window_attributes_predictions = network_obj(test_batch_v)
                #class_predictions = class_predictions[:, range_scores[self.config["classification_target"]]]

                #results_test = metrics_obj.metric(targets=test_batch_l_a_class, predictions=class_predictions)

                window_predictions = window_predictions.reshape(b_windows_sample, n_windows_sample, window_predictions.size()[1])
                window_attributes_predictions = window_attributes_predictions.reshape(b_windows_sample, n_windows_sample,
                                                                window_attributes_predictions.size()[1])

                print(class_predictions[0, :8])

                print(test_batch_v.size()[0])
                for fb in range(test_batch_v.size()[0]):
                    #try:
                    print(fb, test_batch_l_p_class[fb])
                    print(test_batch_l_a_class.size())
                    obj = {
                        "data": pseudo_annotated["data"][fb].to("cpu", torch.double),#.numpy().astype(np.float16),
                        "labels": pseudo_annotated["labels"][fb].to("cpu", torch.double),#.numpy().astype(np.float16),
                        "identity": pseudo_annotated["identity"][fb],
                        "label_file": pseudo_annotated["label_file"][fb]
                        .to("cpu", torch.double)
                        .numpy()
                        .astype(np.float16),
                        #"attrs_predictions": attrs_predictions[fb].to("cpu", torch.double).numpy().astype(np.float16),
                        "overall_predictions": class_predictions[fb].to("cpu", torch.double),#.numpy().astype(np.float16),
                        "window_predictions": window_predictions[fb].to("cpu", torch.double),#.numpy().astype(np.float16),
                        "window_attributes_predictions": window_attributes_predictions[fb].to("cpu", torch.double),
                        # .numpy().astype(np.float16),
                        "test_batch_l_p_class": test_batch_l_p_class[fb].to("cpu", torch.int8),
                        "test_batch_l_a_class": test_batch_l_a_class.to("cpu", torch.int8),
                        "test_batch_l_attribute": test_batch_l_attribute.to("cpu", torch.int8),
                    }

                    print(harwindow_batched_test["file_name"][fb])
                    new_file = harwindow_batched_test["file_name"][fb].replace(
                        "/mnt/data/femo/Documents/CAR/Segmented_windows/mm_car/sequences_test/",
                        "/mnt/data/femo/Documents/CAR/Segmented_windows/mm_car/sequences_evaluated/",
                    )
                    print(new_file)
                    file_name = open(new_file, "wb")

                    pickle.dump(obj, file_name, protocol=pickle.HIGHEST_PROTOCOL)

                    sys.stdout.write(
                        "\r" + "Creating sequence file number {}".format(harwindow_batched_test["file_name"][fb])
                    )
                    sys.stdout.flush()

                    file_name.close()
                    #except:
                    #    raise ("\nError adding the seq {} from {} \n".format(fb, self.config["batch_size_train"]))

                #sys.stdout.write("\rTesting: Batch  {}/{}".format(v, len(dataLoader_test)))
                #sys.stdout.flush()
                print("\rTesting: Batch  {}/{}".format(v, len(dataLoader_test)))

        elapsed_time_test = time.time() - start_time_test

        del test_batch_v
        del class_predictions, window_predictions
        del network_obj

        torch.cuda.empty_cache()

        return

    ##################################################
    ############  evolution_evaluation  ##############
    ##################################################

    def evolution_evaluation(self, ea_iter, testing=False):
        """
        Organises the evolution, training, testing or validating

        @param ea_itera: evolution iteration
        @param testing: Setting testing in training or only testing
        @return results: dict with validating/testing results
        @return confusion_matrix: dict with validating/testing results
        @return best_itera: best iteration for training
        """

        logging.info("        Network_User: Evolution evaluation iter {}".format(ea_iter))

        confusion_matrix = 0
        best_itera = 0
        if testing:
            logging.info("        Network_User: Testing")
            results, confusion_matrix = self.test(ea_iter)
        else:
            if self.config["usage_modus"] == "train":
                logging.info("        Network_User: Training")

                results_train, best_itera = self.train(ea_iter)
                results, _ = self.test(ea_iter, ifVal=True)

            elif self.config["usage_modus"] == "train_final":
                logging.info("        Network_User: Final Training")
                results, best_itera = self.train(ea_iter)
                results, _ = self.test(ea_iter, ifVal=True)

            elif self.config["usage_modus"] == "fine_tuning":
                logging.info("        Network_User: Fine Tuning")
                results, best_itera = self.train(ea_iter)
                #results, _ = self.test(ea_iter, ifVal=True)

            elif self.config["usage_modus"] == "test":
                logging.info("        Network_User: Testing")
                self.pseudo_annotate()
                results, confusion_matrix = self.test(ea_iter, ifVal=False)

            else:
                logging.info("        Network_User: Not selected modus ")

        return results, confusion_matrix, best_itera
