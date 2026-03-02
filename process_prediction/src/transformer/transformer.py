"""
@created: 16.03.2021
@author: Fernando Moya Rueda

@copyright: Motion Miners GmbH, Emil-Figge Str. 76, 44227 Dortmund, 2022

@brief: tCNN for ErgoCom
"""

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        Wx = 500 #self.config["sliding_window_length"]

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

        # MLP
        self.fc4 = nn.Linear(256 * 3, 128)

        #Transformer
        self.window_size = int((self.config["sliding_window_length"] - 500) / 100 + 1)
        self.encode_position = False
        self.transformer_dim = 128
        encoder_layer = TransformerEncoderLayer(d_model=self.transformer_dim,
                                                nhead=8,
                                                dim_feedforward=128,
                                                dropout=0.1,
                                                activation="gelu",
                                                batch_first=True)

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = 6,
                                              norm = nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(1, self.window_size + 1, self.transformer_dim))

        self.norm_layer = nn.LayerNorm(self.transformer_dim)

        self.fc5 = nn.Linear(128, self.config["num_attributes"])

        if self.config["reshape_input"]:
            self.avgpool = nn.AvgPool2d(kernel_size=[1, int(self.num_channels / 3)])
        else:
            self.avgpool = nn.AvgPool2d(kernel_size=[1, self.num_channels])

        # Task for MotionMiners
        # self.ERGOKOM = nn.Linear(self.config['num_attributes'], self.config['num_classes']) #4
        self.process_mp = nn.Linear(128, 11)  # 5
        self.process_lp = nn.Linear(128, 31)  # 8
        self.activities = nn.Linear(128, 15)  # 27 + 46

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

        if x.get_device() == -1:
            local_device = torch.device("cpu")
        elif x.get_device() == 0:
            local_device = torch.device("cuda:0")

        x_unfold = x.permute(0, 1, 3, 2)  # input [B, 1, T, C] -> [B, 1, C, T]
        x_unfold = x_unfold.unfold(dimension=3, size=500, step=100)  # input [B, 1, C, W, T2]
        x_unfold = x_unfold.permute(0, 3, 1, 4, 2)  # input [B, W, 1, 500, C]
        b_windows_sample = x_unfold.size()[0]
        n_windows_sample = x_unfold.size()[1]

        # input [B x W, 1, 500, C]
        x_unfold = x_unfold.reshape(-1, x_unfold.size()[2], x_unfold.size()[3], x_unfold.size()[4]) #[B X W, 1, 500, 18]

        # Selecting the one ot the two networks, tCNN or tCNN-IMU
        if self.config["network"] == "cnn_imu":
            x_LA, x_N, x_RA = self.tcnn_imu(x_unfold)
            x_u = torch.cat((x_LA, x_N, x_RA), 1) # [B x W, 768]

        # Selecting MLP, FC
        x_u = F.dropout(x_u, training=self.training)
        x_u = F.relu(self.fc4(x_u))
        x_fc4 = F.dropout(x_u, training=self.training) # [B x W, 128]

        x_fc4_unfold = x_fc4.reshape(b_windows_sample, n_windows_sample, x_fc4.size()[1]) #[B, W, 128]


        #Transformer
        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(x_fc4_unfold.shape[0], 1, 1) # [B, 1, 128]
        #print("cls token", cls_token.size())
        x_fc4_cls_token = torch.cat([cls_token, x_fc4_unfold], 1) # [B, W + 1, 128]
        #print("x_fc4 cat clstoken", x_fc4_cls_token.size())
        #print(torch.sum(x_fc4_cls_token[0]))

        # Add the position embedding
        if self.encode_position:
            x_fc4_cls_token += self.position_embed
            #print("encode_position fc4", x_fc4.size())

        # Transformer Encoder pass
        #print(x_fc4_cls_token.size())
        x_fc4_tr = self.transformer_encoder(x_fc4_cls_token)#[0]
        #print("x_fc4 transformer", x_fc4_tr.size())
        #print(torch.sum(x_fc4[0]))
        #print("from 0", x_fc4[0, 0])
        #print("from 1", x_fc4[-1, 0])

        x_fc4_tr = x_fc4_tr[:, 0, :]
        x_fc4_tr = self.norm_layer(x_fc4_tr)


        if self.training:
            attr = self.fc5(x_fc4_tr)
            attr = self.sigmoid(attr)
            process_mp = self.process_mp(x_fc4_tr)
            process_lp = self.process_lp(x_fc4_tr)
            activities = self.activities(x_fc4_tr)

        elif not self.training:
            attr = self.fc5(x_fc4_tr)
            attr = self.sigmoid(attr)

            process_mp = self.process_mp(x_fc4_tr)
            process_mp = self.softmax(process_mp)

            process_lp = self.process_lp(x_fc4_tr)
            process_lp = self.softmax(process_lp)

            # HANDLING_HEIGHTS = attr[:, [3, 4, 5]]
            activities = self.activities(x_fc4_tr)
            activities = self.softmax(activities)

        x = torch.cat(
            (
                process_mp,
                process_lp,
                activities
            ),
            1,
        )

        return x

    def init_weights(self):
        """
        Applying initialisation of layers
        """
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
