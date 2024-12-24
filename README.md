# Hierarchical GeoCLIP

The project implements GeoCLIP, a CLIP-inspired image geolocation model. Hierarchical GeoCLIP is an improved inference method that leverages hierarchical feature clustering at multiple geographical scales. By organizing the GPS gallery into a tree structure, we drastically reduce the search space and achieve comparable performance while being ~100x more efficient. 

# Instructions to run

1.  Clone the repository and go to the project folder.
```
git clone https://github.com/ramanakshay/hierarchical-geoclip
cd hierarchical-geoclip
```

2. Install dependencies from requirements file. Make sure to create a virtual environment before running this command.
```
pip install -r requirements.txt
```

4. Download dataset from scripts inside the data folder. Run main.py to train the model
```
python main.py
```

## Requirements
- [pytorch](https://pytorch.org/) (An open source deep learning platform)
- [hydra](https://hydra.cc/) (A framework for configuring complex applications)
- [sklearn]() (For clustering algorithms)
- [datasets]() (Hugging Face Datasets)


# Folder Structure
```
├──  model              - this folder contains all code (models, networks) of the agent
│   ├── classifier.py
│   ├── network.py
│   └── layers.py
│
│
├── data                - this folder contains code relevant to the data
│   ├── preprocessing.py         - run this script to download the data   
│   └── dataset.py             - dataset classes

│
│
├── algorithm             - this folder contains different algorithms of your project
│   └── train.py
│
│
├──  config
│    └── config.yaml  - YAML config file for project
│
│
├──  clustering            - this folder contains code for clustering the GPS gallery
│    └── cluster.py        - build JSON file from list of GPS coordinates and pre-trained model
│
│
└── main.py           - entry point of the project

```

# License

This project is licensed under the MIT License. See LICENSE for more details.
