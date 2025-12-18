# Process Prediction

Here, you will find the code for predicting processes from time series data containing IMU data and /or BLE RSSI data.
The goal is to predict low-level, mid-level and high-level processes from the time series data.
The code is structured as follows.

Sequences of data from subjects are segmented into Process Windows Np windows of size Tp. So, each segmented window will 
be of size [Tp, S] for S number of features. These windows will be intended for mid-level prediction.

## Low Level Prediction
These Process Windows will be then processed by 3 process predictors. For predicting, first, a single Process Window 
[1, Tp, S] is segmented into N_lp low-level windows of size T_lp. So each Process Window [1, Tp, S] is segmented 
into -> [1, N_lp, T_lp, S], sequence of low-level windows. Second, the predictor will map sequence of low-level windows 
into a sequence of low-level predictions -> [1, N_lp, T_lp, K]. 

## Mid Level Prediction

These Process Windows are then processed by 3 process predictors. First, a Process Windows [Np, Tp, K] is segmented into
N_lp low-level windows of size T_lp, with K the one-hot encoding of the low-level process. So each Process Window [1, Tp, S] is segmented into -> [1, N_lp, T_lp, S], sequence 
of low-level windows. Second, the predictor will map sequence of low-level windows into a sequence of low-level 
predictions -> [1, N_lp, K]. 


We will have 3 methods for process predictions:
- Transformer
- LSTM
- HMM-GMM# Process Prediction

Here, you will find the code for predicting processes from time series data containing IMU data and /or BLE RSSI data.
The goal is to predict low-level, mid-level and high-level processes from the time series data.
The code is structured as follows.

Sequences of data from subjects are segmented into Process Windows Np windows of size Tp. So, each segmented window will be of size 
[Tp, S] for S number of features. These windows will be intended for mid-level prediction.

These Process Windows are then processed by 3 process predictors. First, a Process Windows [Np, Tp, S] is segmented into
N_lp low-level windows of size T_lp. So each Process Window [1, Tp, S] is segmented into -> [1, N_lp, T_lp, S], sequence of 
low-level windows. Second, the predictor will map sequence of low-level windows into a sequence of low-level predictions
-> [1, N_lp, K]

We will have 3 methods for process predictions:
- Transformer
  - This network will map a sequence input, either a seque into an output window
- LSTM
- HMM-GMM

For training, randomized Process Windows, are 


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



For training, randomized Process Windows, are 


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

