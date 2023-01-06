# V-WSD

This repository is for CLIP fine-tuning.

## Directory
+ **main/res** - Resources including model check points, datasets, and experiment records
+ **main/src** - Source code including model structures, training pipelines, and utility functions
+ **res/data** - Need to place the data folder under here
+ **res/models** - Need to place the clip-vit-base-patch32 repo under here
```
Mask-Controlled-Paraphrase-Generation
├── README.md
├── main
│   ├── config.py
│   ├── main.py
│   ├── res
│   │   ├── ckpts
│   │   ├── data
│   │   │   └── semeval-2023-task-1-V-WSD-train-v1
│   │   ├── models
│   │   │   └── clip-vit-base-patch32
│   │   └── log
│   └── src
│         ├── datasets.py
│         ├── trainers.py
│         └── utils
│             └── helper.py
└── requirements.txt
```

## Dependencies
+ python >= 3.10.8
+ torch >= 1.13.1
+ torchvision >= 0.14.1
+ transformers >= 4.25.1

## Setup
Please ensure required packages are already installed. A virtual environment is recommended.
```
$ cd clip
$ pip install pip --upgrade
$ pip install -r requirements.txt
$ cd main
$ mkdir res
$ cd res
$ mkdir models
$ cd models
$ git clone https://huggingface.co/openai/clip-vit-base-patch32
$ cd ../..
$ python main.py
```

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com