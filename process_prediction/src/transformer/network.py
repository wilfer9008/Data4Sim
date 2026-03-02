"""
@created: 16.03.2021
@author: Fernando Moya Rueda

@copyright: Motion Miners GmbH, Emil-Figge Str. 76, 44227 Dortmund, 2022

@brief: tCNN for ErgoCom
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Network(nn.Module):
    """
    classdocs
    """

    def __init__(self, config):
        """
        Constructor
        """

        super(Network, self).__init__()

        logging.info("            Network: Constructor")

        self.config = config
        self.num_channels = self.config["NB_sensor_channels"]

        if not self.config["magnetometer"]:
            self.num_channels = self.num_channels - 9

        if self.config["reshape_input"]:
            in_channels = 3
            self.num_channels = self.num_channels / 3
            Hx = int(self.num_channels)
        else:
            in_channels = 1
            Hx = self.num_channels
        Wx = 100 #self.config["sliding_window_length"]  #100

        padding = 0

        # Computing the size of the feature maps
        Wx, Hx = self.size_feature_map(
            Wx=Wx, Hx=Hx, F=(self.config["filter_size"], 1), P=padding, S=(1, 1), type_layer="conv"
        )
        logging.info("            Network: Wx {} and Hx {}".format(Wx, Hx))
        Wx, Hx = self.size_feature_map(
            Wx=Wx, Hx=Hx, F=(self.config["filter_size"], 1), P=padding, S=(1, 1), type_layer="conv"
        )
        logging.info("            Network: Wx {} and Hx {}".format(Wx, Hx))

        Wx, Hx = self.size_feature_map(
            Wx=Wx, Hx=Hx, F=(self.config["filter_size"], 1), P=padding, S=(1, 1), type_layer="conv"
        )
        logging.info("            Network: Wx {} and Hx {}".format(Wx, Hx))
        Wx, Hx = self.size_feature_map(
            Wx=Wx, Hx=Hx, F=(self.config["filter_size"], 1), P=padding, S=(1, 1), type_layer="conv"
        )
        logging.info("            Network: Wx {} and Hx {}".format(Wx, Hx))

        ######################################
        ##########Transformer#################
        ######################################

        # set the Conv layers
        if self.config["network"] in ["cnn_imu", "cnn_imu_tpp"]:
            # LA
            self.Trans_conv_LA_1_1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_LA_1_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_LA_2_1 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_LA_2_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            if self.config["reshape_input"]:
                if not self.config["magnetometer"]:
                    self.Trans_fc3_LA = nn.Linear(
                        self.config["num_filters"] * int(Wx) * int(self.config["NB_sensor_channels"] / 6), 256
                    )
                else:
                    self.Trans_fc3_LA = nn.Linear(
                        self.config["num_filters"] * int(Wx) * int(self.config["NB_sensor_channels"] / 9), 256
                    )
            else:
                self.Trans_fc3_LA = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 3), 256)

            # N
            self.Trans_conv_N_1_1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_N_1_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_N_2_1 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_N_2_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            if self.config["reshape_input"]:
                if not self.config["magnetometer"]:
                    self.Trans_fc3_N = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 6), 256)
                else:
                    self.Trans_fc3_N = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 9), 256)
            else:
                self.Trans_fc3_N = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 3), 256)

            # RA
            self.Trans_conv_RA_1_1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_RA_1_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_RA_2_1 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_RA_2_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            if self.config["reshape_input"]:
                if not self.config["magnetometer"]:
                    self.Trans_fc3_RA = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 6), 256)
                else:
                    self.Trans_fc3_RA = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 9), 256)

            else:
                self.Trans_fc3_RA = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 3), 256)

            # LL
            self.Trans_conv_LL_1_1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_LL_1_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_LL_2_1 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_LL_2_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            if self.config["NB_sensor_channels"] != 27:
                if self.config["reshape_input"]:
                    if not self.config["magnetometer"]:
                        self.Trans_fc3_LL = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 6), 256)
                    else:
                        self.Trans_fc3_LL = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 9), 256)
                else:
                    self.Trans_fc3_LL = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 3), 256)

            # RL
            self.Trans_conv_RL_1_1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_RL_1_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_RL_2_1 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.Trans_conv_RL_2_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            if self.config["NB_sensor_channels"] != 27:
                if self.config["reshape_input"]:
                    if not self.config["magnetometer"]:
                        self.Trans_fc3_RL = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 6), 256)
                    else:
                        self.Trans_fc3_RL = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 9), 256)
                else:
                    self.Trans_fc3_RL = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 3), 256)

        ######################################
        ##########TCNN#################
        ######################################

        Wx = 150 #self.config["sliding_window_length"]  #100

        padding = 0

        # Computing the size of the feature maps
        Wx, Hx = self.size_feature_map(
            Wx=Wx, Hx=Hx, F=(self.config["filter_size"], 1), P=padding, S=(1, 1), type_layer="conv"
        )
        logging.info("            Network: Wx {} and Hx {}".format(Wx, Hx))
        Wx, Hx = self.size_feature_map(
            Wx=Wx, Hx=Hx, F=(self.config["filter_size"], 1), P=padding, S=(1, 1), type_layer="conv"
        )
        logging.info("            Network: Wx {} and Hx {}".format(Wx, Hx))

        Wx, Hx = self.size_feature_map(
            Wx=Wx, Hx=Hx, F=(self.config["filter_size"], 1), P=padding, S=(1, 1), type_layer="conv"
        )
        logging.info("            Network: Wx {} and Hx {}".format(Wx, Hx))
        Wx, Hx = self.size_feature_map(
            Wx=Wx, Hx=Hx, F=(self.config["filter_size"], 1), P=padding, S=(1, 1), type_layer="conv"
        )
        logging.info("            Network: Wx {} and Hx {}".format(Wx, Hx))

        # set the Conv layers
        if self.config["network"] in ["cnn_imu", "cnn_imu_tpp"]:
            # LA
            self.conv_LA_1_1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_LA_1_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_LA_2_1 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_LA_2_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            if self.config["reshape_input"]:
                if not self.config["magnetometer"]:
                    self.fc3_LA = nn.Linear(
                        self.config["num_filters"] * int(Wx) * int(self.config["NB_sensor_channels"] / 6), 256
                    )
                else:
                    self.fc3_LA = nn.Linear(
                        self.config["num_filters"] * int(Wx) * int(self.config["NB_sensor_channels"] / 9), 256
                    )
            else:
                self.fc3_LA = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 3), 256)

            # N
            self.conv_N_1_1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_N_1_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_N_2_1 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_N_2_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            if self.config["reshape_input"]:
                if not self.config["magnetometer"]:
                    self.fc3_N = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 6), 256)
                else:
                    self.fc3_N = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 9), 256)
            else:
                self.fc3_N = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 3), 256)

            # RA
            self.conv_RA_1_1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_RA_1_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_RA_2_1 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_RA_2_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            if self.config["reshape_input"]:
                if not self.config["magnetometer"]:
                    self.fc3_RA = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 6), 256)
                else:
                    self.fc3_RA = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 9), 256)

            else:
                self.fc3_RA = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 3), 256)

            # LL
            self.conv_LL_1_1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_LL_1_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_LL_2_1 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_LL_2_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            if self.config["NB_sensor_channels"] != 27:
                if self.config["reshape_input"]:
                    if not self.config["magnetometer"]:
                        self.fc3_LL = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 6), 256)
                    else:
                        self.fc3_LL = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 9), 256)
                else:
                    self.fc3_LL = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 3), 256)

            # RL
            self.conv_RL_1_1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_RL_1_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_RL_2_1 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            self.conv_RL_2_2 = nn.Conv2d(
                in_channels=self.config["num_filters"],
                out_channels=self.config["num_filters"],
                kernel_size=(self.config["filter_size"], 1),
                stride=1,
                padding=padding,
            )

            if self.config["NB_sensor_channels"] != 27:
                if self.config["reshape_input"]:
                    if not self.config["magnetometer"]:
                        self.fc3_RL = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 6), 256)
                    else:
                        self.fc3_RL = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 9), 256)
                else:
                    self.fc3_RL = nn.Linear(self.config["num_filters"] * int(Wx) * int(self.num_channels / 3), 256)

        # MLP
        self.Trans_fc4 = nn.Linear(256 * 3, 256)
        self.fc4 = nn.Linear(256 * 3, 256)
        self.fc5 = nn.Linear(86*24, 24)
        self.fc5_old_windows = nn.Linear(100 * 24, 24)
        self.Trans_fc4_2 = nn.Linear(self.config["num_attributes"] + 14, self.config["num_attributes"] + 14)

        #Recurrent
        self.fc4_R = nn.LSTM(input_size=self.config["num_attributes"] + 14, hidden_size=self.config["num_attributes"] + 14,
                             batch_first=True, bidirectional=False)

        #Transformer
        #self.window_size = int((self.config["sliding_window_length"] - 100) / 10 + 1)
        #self.encode_position = False
        #self.transformer_dim = 256
        #encoder_layer = TransformerEncoderLayer(d_model=self.transformer_dim,
        #                                        nhead=8,
        #                                        dim_feedforward=256,
        #                                        dropout=0.1,
        #                                        activation="gelu",
        #                                        batch_first=True)

        #self.transformer_encoder = TransformerEncoder(encoder_layer,
        #                                      num_layers = 6,
        #                                      norm = nn.LayerNorm(self.transformer_dim))
        #self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        #if self.encode_position:
        #self.position_embed = nn.Parameter(torch.randn(1, self.window_size + 1, self.transformer_dim))

        #self.norm_layer = nn.LayerNorm(self.transformer_dim)

        # if self.config['output'] == 'softmax':
        #    self.fc5 = nn.Linear(256, self.config['num_classes'])
        # elif self.config['output'] == 'attribute':
        self.Trans_fc5 = nn.Linear(self.config["num_attributes"] + 24 + 14, self.config["num_attributes"])
        self.fc5_activity = nn.Linear(256 + 24, self.config["num_classes"])
        self.fc5_attrs_activity = nn.Linear(256 + 24, self.config["num_attributes"])
        # self.fc5 = nn.Linear(256, 19)

        if self.config["reshape_input"]:
            self.avgpool = nn.AvgPool2d(kernel_size=[1, int(self.num_channels / 3)])
        else:
            self.avgpool = nn.AvgPool2d(kernel_size=[1, self.num_channels])

        # Task for MotionMiners
        # self.ERGOKOM = nn.Linear(self.config['num_attributes'], self.config['num_classes']) #4
        self.PROCESS = nn.Linear(self.config["num_attributes"] + 24 + 14, 8)  # 5
        self.ACTIVITY = nn.Linear(self.config["num_attributes"] + 24 + 14, 14)  # 8

        self.BASE_ACTIVITIES = nn.Linear(256, 5)  # 5
        self.HANDLING_HEIGHTS = nn.Linear(256, 3)  # 8
        self.WRAP = nn.Linear(256, 2)  # 10 +16
        self.LARA = nn.Linear(256, 7)  # 23
        self.PULL_PUSH = nn.Linear(256, 2)  # 25
        self.WRITE = nn.Linear(256, 2)  # 27 + 46
        self.AIRBUS = nn.Linear(256, 3)  # 3

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        return

    def forward(self, x):
        """
        Forwards function, required by torch.

        @param x: batch [batch, 1, Time, Channels], Channels = Sensors * 3 Axis
        @return x: Output of the network, either Softmax or Attribute
        """

        # As magnetometer is still prsent in the IMUs, they are ignored
        if x.size()[3] == 27 and not self.config["magnetometer"]:
            idx_without_mag = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23]
            x = x[:, :, :, idx_without_mag]

        # TCNN input [B, 1, T, C]
        #print("input: ", x.size())

        if x.get_device() == -1:
            local_device = torch.device("cpu")
        elif x.get_device() == 0:
            local_device = torch.device("cuda:0")

        x_unfold = x.permute(0, 1, 3, 2)  # input [B, 1, T, C] -> [B, 1, C, T]
        x_unfold = x_unfold.unfold(dimension=3, size=150, step=10)  # input [B, 1, C, W, T]
        x_unfold = x_unfold.permute(0, 3, 1, 4, 2)  # input [B, W, 1, 150, C]
        b_windows_sample = x_unfold.size()[0]
        n_windows_sample = x_unfold.size()[1]

        # input [B x W, 1, 150, C]
        x_unfold = x_unfold.reshape(-1, x_unfold.size()[2], x_unfold.size()[3], x_unfold.size()[4])

        # Selecting the one ot the two networks, tCNN or tCNN-IMU
        if self.config["network"] == "cnn_imu":
            x_LA_u, x_N_u, x_RA_u = self.tcnn_imu(x_unfold)
            x_u = torch.cat((x_LA_u, x_N_u, x_RA_u), 1)

        # Selecting MLP, FC
        x_u = F.dropout(x_u, training=self.training) # [B x W, 256 x 3]
        x_u = F.relu(self.fc4(x_u)) # [B x W, 256]
        x_tcnn_fc4 = F.dropout(x_u, training=self.training)

        BASE_ACTIVITIES = self.BASE_ACTIVITIES(x_tcnn_fc4) # [430, 5]
        BASE_ACTIVITIES = self.softmax(BASE_ACTIVITIES)
        HANDLING_HEIGHTS = self.HANDLING_HEIGHTS(x_tcnn_fc4) # [430, 3]
        HANDLING_HEIGHTS = self.softmax(HANDLING_HEIGHTS)
        WRAP = self.WRAP(x_tcnn_fc4) # [430, 2]
        WRAP = self.softmax(WRAP)
        LARA = self.LARA(x_tcnn_fc4) # [430, 7]
        LARA = self.softmax(LARA)
        PULL_PUSH = self.PULL_PUSH(x_tcnn_fc4) # [430, 2]
        PULL_PUSH = self.softmax(PULL_PUSH)
        WRITE = self.WRITE(x_tcnn_fc4) # [430, 2]
        WRITE = self.softmax(WRITE)
        AIRBUS = self.AIRBUS(x_tcnn_fc4) # [430, 3]
        AIRBUS = self.softmax(AIRBUS)

        old_classification = torch.cat((BASE_ACTIVITIES, HANDLING_HEIGHTS, WRAP, LARA, PULL_PUSH, WRITE, AIRBUS), 1)  # [430, 24]
        old_classification_fold = old_classification.reshape(b_windows_sample, n_windows_sample, 24)  # [B, W, 24]

        old_classification_fold_h = torch.empty((b_windows_sample, 1000, 24), device=local_device) #[B, 1000, 24]
        for b in range(b_windows_sample):
            old_classification_fold_h[b] = self.unsegmenting(predictions=old_classification_fold[b],
                                                        size_samples=1000,
                                                        window_length=150,
                                                        step=10,
                                                        num_classes=24,
                                                        device=local_device)
        old_classification_fold = old_classification_fold_h.unfold(dimension=1, size=100, step=10)  # input [B, 91, 24, 100]
        old_classification_fold = old_classification_fold.permute(0, 1, 3, 2)  # input [B, W, 100, 24]
        old_classification_fold = old_classification_fold.reshape(-1, old_classification_fold.size()[2],  old_classification_fold.size()[3]) #input [B x W, 100, 24]
        old_classification_fold = old_classification_fold.reshape(old_classification_fold.size()[0], old_classification_fold.size()[1] * old_classification_fold.size()[2]) # [B x W, 2400]

        old_classification_fold = self.fc5_old_windows(old_classification_fold) # [B x W, 24]
        old_classification_fold = F.relu(old_classification_fold)

        old_classification = old_classification.reshape(b_windows_sample, n_windows_sample, 24) # [B, W, 24] # [B, 86, 24]
        old_classification = old_classification.reshape(-1, old_classification.size()[1] * old_classification.size()[2]) # [B, W * 24]
        old_classification = self.fc5(old_classification) #[B, 24]
        old_classification = F.relu(old_classification)

        # 2nd branch input [B, 1, T, C]
        x_unfold = x.permute(0, 1, 3, 2)  # input [B, 1, C, T]
        x_unfold = x_unfold.unfold(dimension=3, size=100, step=10)  # input [B, 1, C, W, 100]
        x_unfold = x_unfold.permute(0, 3, 1, 4, 2)  # input [B, W, 1, 100, C]
        b_windows_sample = x_unfold.size()[0]
        n_windows_sample = x_unfold.size()[1]

        # input [B x W, 1, 100, C]
        x_unfold = x_unfold.reshape(-1, x_unfold.size()[2], x_unfold.size()[3], x_unfold.size()[4])

        # Selecting the one ot the two networks, tCNN or tCNN-IMU
        if self.config["network"] == "cnn_imu":
            x_LA_u, x_N_u, x_RA_u = self.tcnn_trans_imu(x_unfold)
            x_u = torch.cat((x_LA_u, x_N_u, x_RA_u), 1)

        # Selecting MLP, FC
        x_u = F.dropout(x_u, training=self.training) #[BxW, 256x3]
        x_u = F.relu(self.Trans_fc4(x_u)) #[BxW, 256]
        x_fc4 = F.dropout(x_u, training=self.training)

        x_fc4 = torch.cat((x_fc4, old_classification_fold), 1) # [BxW, 256] + [BxW, 256] -> [BxW, 280]

        # input [B x W, 6, C]
        x_fc5_windows = self.fc5_activity(x_fc4) #[B * W, 14]
        x_fc5_windows = self.softmax(x_fc5_windows)
        x_fc5_attrs_windows = self.fc5_attrs_activity(x_fc4) #[B * W, 63]
        x_fc5_attrs_windows = self.sigmoid(x_fc5_attrs_windows)

        x_fc4_windows_concat = torch.cat((x_fc5_windows, x_fc5_attrs_windows), 1) #[B x W, 14 + 63]

        # [B, W, 77]
        x_fc4_unfold = x_fc4_windows_concat.reshape(b_windows_sample, n_windows_sample, x_fc4_windows_concat.size()[1])

        x_fc4_R = self.fc4_R(x_fc4_unfold)[0] #[B, W, 77]
        x_fc4_R = F.relu(x_fc4_R)
        x_fc4_R = x_fc4_R + x_fc4_unfold # [B, W, 77] + [B, W, 77] -> [B, W, 77]

        #Transformer
        # Prepend class token
        #cls_token = self.cls_token.unsqueeze(1).repeat(x_fc4_unfold.shape[0], 1, 1)
        #print("cls token", cls_token.size())
        #x_fc4_cls_token = torch.cat([cls_token, x_fc4_unfold], 1)
        #print("x_fc4 cat clstoken", x_fc4_cls_token.size())
        #print(torch.sum(x_fc4_cls_token[0]))

        # Add the position embedding
        #if self.encode_position:
        #    x_fc4_cls_token += self.position_embed
            #print("encode_position fc4", x_fc4.size())

        # Transformer Encoder pass
        #print(x_fc4_cls_token.size())
        #x_fc4_tr = self.transformer_encoder(x_fc4_cls_token)#[0]
        #print("x_fc4 transformer", x_fc4_tr.size())
        #print(torch.sum(x_fc4[0]))
        #print("from 0", x_fc4[0, 0])
        #print("from 1", x_fc4[-1, 0])

        #x_fc4_tr = x_fc4_tr[:, 0, :]
        #x_fc4_tr = self.norm_layer(x_fc4_tr)

        x_fc4 = self.Trans_fc4_2(x_fc4_R[:, -1, :]) #[B, -1, 77] -> #[B, 77]
        #x_fc4 = self.fc4_2(x_fc4)
        x_fc4 = F.relu(x_fc4)

        x_fc4 = torch.cat((x_fc4, old_classification), 1) #[B, 101]

        if self.training:
            attr = self.Trans_fc5(x_fc4) #[B, 63]
            attr = self.sigmoid(attr)

            PROCESS = self.PROCESS(x_fc4) #[B, 8]
            ACTIVITY = self.ACTIVITY(x_fc4) #[B, 14]

            #x_fc5_attrs_windows = self.sigmoid(x_fc5_attrs_windows)

        elif not self.training:
            attr = self.Trans_fc5(x_fc4)
            attr = self.sigmoid(attr)

            PROCESS = self.PROCESS(x_fc4)
            PROCESS = self.softmax(PROCESS)

            ACTIVITY = self.ACTIVITY(x_fc4)
            ACTIVITY = self.softmax(ACTIVITY)

        x = torch.cat(
            (
                PROCESS,
                ACTIVITY,
                attr,
            ),
            1,
        )

        return x, x_fc5_windows, x_fc5_attrs_windows

    def init_weights(self):
        """
        Applying initialisation of layers
        """
        if True:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.apply(Network._init_weights_orthonormal)
        return

    @staticmethod
    def _init_weights_orthonormal(m):
        """
        Orthonormal Initialissation of layer

        @param m: layer m
        """
        if isinstance(m, nn.Conv2d):
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias.data, 0)
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias.data, 0)

        return

    def size_feature_map(self, Wx, Hx, F, P, S, type_layer="conv"):
        """
        Computing size of feature map after convolution or pooling

        @param Wx: Width input
        @param Hx: Height input
        @param F: Filter size
        @param P: Padding
        @param S: Stride
        @param type_layer: conv or pool
        @return Wy: Width output
        @return Hy: Height output
        """

        if self.config["aggregate"] in ["FCN", "LSTM"]:
            Pw = P[0]
            Ph = P[1]
        elif self.config["aggregate"] == "FC":
            Pw = P
            Ph = P

        if type_layer == "conv":
            Wy = 1 + (Wx - F[0] + 2 * Pw) / S[0]
            Hy = 1 + (Hx - F[1] + 2 * Ph) / S[1]

        elif type_layer == "pool":
            Wy = 1 + (Wx - F[0]) / S[0]
            Hy = 1 + (Hx - F[1]) / S[1]

        return Wy, Hy

    def unsegmenting(self, predictions, size_samples, window_length, step, num_classes, device):
        """
        Mapping window-wise predictions to sample-wise predictions with an overlapping given by Step.
        It computes the mean over the accumulated probabilities per sample in time divided by the counter
        of how many windows fall into the sample
        Args:
            predictions: Window-wise predictions from the network
            size_samples: number of windows
            window_length: number of samples per window, e.g., 150 samples  = 1.5 seconds
            step: Step made for window-wise segmentation, e.e.g, 25 samples = 12.5% overlapping
            device: The device where the unsegmeting should be calculated (e.g. torch.device('cuda:0'))

        Returns: Unsegmented Data of shape [time x classes]

        """
        sample_predictions = torch.zeros((num_classes, size_samples), device=device)
        sample_predictions_prop = torch.zeros((num_classes, size_samples), device=device)
        for ws in range(predictions.size(0)):
            # Creates a tensor with [window_length x classes] shape with the populated with the values the probabilities
            # out of the network
            window_samples = torch.ones((window_length, num_classes), device=device)
            #print(device)
            #print(window_samples.get_device())
            #print(predictions.get_device())
            window_samples = window_samples * predictions[ws]

            # Accumulates the probabilities [window_length x classes] tensor on a accumulator-matrix of shape
            # of the raw recordings [max recording time x classes]
            window_samples = torch.transpose(window_samples, 0, 1)
            range_2_transfer = sample_predictions[:, step * ws: (step * ws) + window_length].size(1)
            sample_predictions[:, step * ws: (step * ws) + window_length] += window_samples[:, :range_2_transfer]

            # Counts the appearance of windows for a certain sample in time
            window_samples_prop = torch.ones((window_length, num_classes), device=device)
            window_samples_prop = torch.transpose(window_samples_prop, 0, 1)
            range_2_transfer = sample_predictions_prop[:, step * ws: (step * ws) + window_length].size(1)
            sample_predictions_prop[:, step * ws: (step * ws) + window_length] += window_samples_prop[
                                                                                  :, :range_2_transfer
                                                                                  ]

        # Computing the mean over the accumulated probabilities divided by the counter
        sample_predictions = torch.div(sample_predictions, sample_predictions_prop)

        # Reshaping to have the same size as the recordings from the sensors [time x classes]
        sample_predictions = torch.transpose(sample_predictions, 0, 1)

        #print("sample_predictions: ", sample_predictions.size())
        return sample_predictions

    def tcnn_imu(self, x):
        """
        tCNN-IMU network
        The parameters will adapt according to the dataset, reshape and output type

        x_LA, x_N, x_RA

        @param x: input sequence
        @return x_LA: Features from left arm
        @return x_N: Features from Neck or Torso
        @return x_RA: Features from Right Arm
        """
        # LA
        if self.config["reshape_input"]:
            if not self.config["magnetometer"]:
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, 0:2]))
            else:
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, 0:3]))
        else:
            if not self.config["magnetometer"]:
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, 0:6]))
            else:
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, 0:9]))
        x_LA = F.relu(self.conv_LA_1_2(x_LA))
        x_LA = F.relu(self.conv_LA_2_1(x_LA))
        x_LA = F.relu(self.conv_LA_2_2(x_LA))
        # view is reshape
        x_LA = x_LA.reshape(-1, x_LA.size()[1] * x_LA.size()[2] * x_LA.size()[3])

        x_LA = F.relu(self.fc3_LA(x_LA))

        # N
        if self.config["reshape_input"]:
            if not self.config["magnetometer"]:
                x_N = F.relu(self.conv_N_1_1(x[:, :, :, 2:4]))
            else:
                x_N = F.relu(self.conv_N_1_1(x[:, :, :, 3:6]))
        else:
            if not self.config["magnetometer"]:
                x_N = F.relu(self.conv_N_1_1(x[:, :, :, 6:12]))
            else:
                x_N = F.relu(self.conv_N_1_1(x[:, :, :, 9:18]))
        x_N = F.relu(self.conv_N_1_2(x_N))
        x_N = F.relu(self.conv_N_2_1(x_N))
        x_N = F.relu(self.conv_N_2_2(x_N))

        # view is reshape
        x_N = x_N.reshape(-1, x_N.size()[1] * x_N.size()[2] * x_N.size()[3])
        x_N = F.relu(self.fc3_N(x_N))

        # RA
        if self.config["reshape_input"]:
            if not self.config["magnetometer"]:
                x_RA = F.relu(self.conv_RA_1_1(x[:, :, :, 4:6]))
            else:
                x_RA = F.relu(self.conv_RA_1_1(x[:, :, :, 6:9]))

        else:
            if not self.config["magnetometer"]:
                x_RA = F.relu(self.conv_RA_1_1(x[:, :, :, 12:18]))
            else:
                x_RA = F.relu(self.conv_RA_1_1(x[:, :, :, 18:27]))

        x_RA = F.relu(self.conv_RA_1_2(x_RA))
        x_RA = F.relu(self.conv_RA_2_1(x_RA))
        x_RA = F.relu(self.conv_RA_2_2(x_RA))

        # view is reshape
        x_RA = x_RA.reshape(-1, x_RA.size()[1] * x_RA.size()[2] * x_RA.size()[3])
        x_RA = F.relu(self.fc3_RA(x_RA))

        return x_LA, x_N, x_RA


    def tcnn_trans_imu(self, x):
        """
        tCNN-IMU network
        The parameters will adapt according to the dataset, reshape and output type

        x_LA, x_N, x_RA

        @param x: input sequence
        @return x_LA: Features from left arm
        @return x_N: Features from Neck or Torso
        @return x_RA: Features from Right Arm
        """
        # LA
        if self.config["reshape_input"]:
            if not self.config["magnetometer"]:
                x_LA = F.relu(self.Trans_conv_LA_1_1(x[:, :, :, 0:2]))
            else:
                x_LA = F.relu(self.Trans_conv_LA_1_1(x[:, :, :, 0:3]))
        else:
            if not self.config["magnetometer"]:
                x_LA = F.relu(self.Trans_conv_LA_1_1(x[:, :, :, 0:6]))
            else:
                x_LA = F.relu(self.Trans_conv_LA_1_1(x[:, :, :, 0:9]))
        x_LA = F.relu(self.Trans_conv_LA_1_2(x_LA))
        x_LA = F.relu(self.Trans_conv_LA_2_1(x_LA))
        x_LA = F.relu(self.Trans_conv_LA_2_2(x_LA))
        # view is reshape
        x_LA = x_LA.reshape(-1, x_LA.size()[1] * x_LA.size()[2] * x_LA.size()[3])

        x_LA = F.relu(self.Trans_fc3_LA(x_LA))

        # N
        if self.config["reshape_input"]:
            if not self.config["magnetometer"]:
                x_N = F.relu(self.Trans_conv_N_1_1(x[:, :, :, 2:4]))
            else:
                x_N = F.relu(self.Trans_conv_N_1_1(x[:, :, :, 3:6]))
        else:
            if not self.config["magnetometer"]:
                x_N = F.relu(self.Trans_conv_N_1_1(x[:, :, :, 6:12]))
            else:
                x_N = F.relu(self.Trans_conv_N_1_1(x[:, :, :, 9:18]))
        x_N = F.relu(self.Trans_conv_N_1_2(x_N))
        x_N = F.relu(self.Trans_conv_N_2_1(x_N))
        x_N = F.relu(self.Trans_conv_N_2_2(x_N))

        # view is reshape
        x_N = x_N.reshape(-1, x_N.size()[1] * x_N.size()[2] * x_N.size()[3])
        x_N = F.relu(self.Trans_fc3_N(x_N))

        # RA
        if self.config["reshape_input"]:
            if not self.config["magnetometer"]:
                x_RA = F.relu(self.Trans_conv_RA_1_1(x[:, :, :, 4:6]))
            else:
                x_RA = F.relu(self.Trans_conv_RA_1_1(x[:, :, :, 6:9]))

        else:
            if not self.config["magnetometer"]:
                x_RA = F.relu(self.Trans_conv_RA_1_1(x[:, :, :, 12:18]))
            else:
                x_RA = F.relu(self.Trans_conv_RA_1_1(x[:, :, :, 18:27]))

        x_RA = F.relu(self.Trans_conv_RA_1_2(x_RA))
        x_RA = F.relu(self.Trans_conv_RA_2_1(x_RA))
        x_RA = F.relu(self.Trans_conv_RA_2_2(x_RA))

        # view is reshape
        x_RA = x_RA.reshape(-1, x_RA.size()[1] * x_RA.size()[2] * x_RA.size()[3])
        x_RA = F.relu(self.Trans_fc3_RA(x_RA))

        return x_LA, x_N, x_RA