# setit2022

Repository for the paper "A machine learning approach for groundwater modeling"


## Installation

Clone this repository

```
git clone https://github.com/galatolofederico/setit2022.git && cd setit2022
```

Create a virtualenv and install the requirements

```
virtualenv --python=python3.7 env
. ./env/bin/activate
pip install -r requirements.txt
```

Download the dataset

```
wget http://131.114.50.176/owncloud/s/PsNS98xzREMseSP/download -O ./dataset.zip
unzip dataset.zip
```

## Usage

To train a model run 

```
python train.py
```

To train a model using the best hyperparameter from the paper

```
python train.py --lr 0.0015901865020523306 --batch-size 32 --patience 50 --loss L1Loss --features-size 49 --encoder-l1-channels 10 --encoder-l2-channels 20 --decoder-l1-channels 20 --decoder-l2-channels 20
```


## Hyperparameter optimization

To run the hyperparameter optimization run

```
easyopt create setit2022-study-1
```

And run as many agents as you want with

```
easyopt agent setit2022-study-1
```


## Build dataset

If you want to re-build the dataset from the raw data run

```
rm -rf ./dataset/dataset

python -m scripts.process-raw-dataset
python -m scripts.build-dataset
python -m scripts.split-dataset
python -m scripts.compute-train-stats
```

## Contributions and license

The code is released as Free Software under the [GNU/GPLv3](https://choosealicense.com/licenses/gpl-3.0/) license. Copying, adapting and republishing it is not only allowed but also encouraged. 

For any further question feel free to reach me at  [federico.galatolo@ing.unipi.it](mailto:federico.galatolo@ing.unipi.it) or on Telegram  [@galatolo](https://t.me/galatolo)