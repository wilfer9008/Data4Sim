# Process Prediction

Here, you will find the code for predicting processes from time series data containing IMU data and /or BLE RSSI data.
The goal is to predict low-level, mid-level and high-level processes from the time series data.
The code is structured as follows.

Sequences of data from subjects are segmented into


## Abstract


## Prerequisites
THe Neuron pruning using Maxout units has the following dependencies:
- Caffe
- numpy
- Python


## Example


Main
 └── Config File (experiment attributes)
     └── Modus (experiment dispatcher)
         ├── Transformer (train / val / test)
         ├── LSTM (train / val / test)
         └── HMM + GMM (train / val / test)

Files
├── main.py
├── config/
│   └── experiment_config.json
├── modus/
│   └── modus.py
├── models/
│   ├── transformers/
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── test.py
│   ├── lstm/
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── test.py
│   └── hmm_gmm/
│       ├── train.py
│       ├── validate.py
│       └── test.py
└── README.md

