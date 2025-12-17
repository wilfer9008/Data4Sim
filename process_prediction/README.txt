Main
 └── Config File (experiment attributes)
     └── Modus (experiment dispatcher)
         ├── Transformer (train / val / test)
         ├── LSTM (train / val / test)
         └── HMM + GMM (train / val / test)
.
├── main.py
├── config/
│   └── experiment_config.yaml
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

