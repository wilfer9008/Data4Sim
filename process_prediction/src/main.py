"""
@created: 08.09.2025
@author: Fernando Moya Rueda

@copyright: Motion Miners GmbH, Emil-Figge Str. 76, 44227 Dortmund, 2022

@brief: Main for Proglove HAR training model using config files
"""

import datetime
import logging
import os
import random
import torch
import platform
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from modus_selecter import Modus_Selecter
import json

class Configuration(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.config = None
        return

    def set_config(self, path):

        f = open(path + "datasets.json")
        self.config = json.load(f)
        f.close()

        return

    def setup_experiment_logger(self, logging_level=logging.DEBUG, filename=None):
        # set up the logging
        logging_format = "[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s"

        if filename != None:
            logging.basicConfig(filename=filename, level=logging.DEBUG, format=logging_format, filemode="w")
        else:
            logging.basicConfig(level=logging_level, format=logging_format, filemode="w")

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger

        if logging.getLogger("").hasHandlers():
            logging.getLogger("").handlers.clear()

        logging.getLogger("").addHandler(console)

        return

    def run_experiment(self, dataset):

        config = self.config[dataset]

        if config["tensorboard_bool"]:
            writer = SummaryWriter(config["folder_exp"] + "tensorboard/" + dataset)
        else:
            writer = None
        self.setup_experiment_logger(logging_level=logging.DEBUG, filename=config["folder_exp"] + "logger.txt")

        logging.info("Finished")

        modus = Modus_Selecter(config, exp=writer)

        # Starting process
        modus.net_modus()

        return



# Press the green button in the gutter to run the script.
if __name__ == "__main__":

    #Setting the same RNG seed
    seed = 125
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)

    print(":Python Platform {}".format(platform.python_version()))

    experiment = Configuration()
    experiment.set_config("../configs/")
    experiment.run_experiment("proglove_fall_2025")

    #"/mnt/data/femo/Documednts/CAR/Segmented_windows/mm_car/",

    print("Done")