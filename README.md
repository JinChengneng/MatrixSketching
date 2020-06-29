# Matrix sketching algorithms and its application in MSD
## Introduction
This repository contains documents and codes for Randomized Linear Algebra Algorithms for Very Over-constrained Least-squares Problem and its Application in Composed Year Prediction of Songs. It implements four matrix sketching algorithms, Gaussian Projection, Subsampled Randomized Hadamard Transform(SRHT), Count Sketch and Leverage Score Sampling and applies them to Million Song Dataset.

## Data
The data required is [YearPredictionMSD](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD).
The recommended way to fetch the dataset is downloading from [here](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/). 
If you are using Linux, it's easy to fetch the dataset via following commands.
```
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2
bzip2 -d YearPredictionMSD.bz2
```
Please move the dataset file to the directory ```./data/``` and run ```python processLibSVMData.py``` to convert the raw data to ```.npy``` document type.

## Install and Usage
To run these codes, ```numpy``` and ```pandas``` are required and ```Python 3``` is recommended.

It's easy to run these codes, just adjust the codes in your desired way and run 
```
python sketch.py
```
 Please wait patiently because it does take a long time to finish. If you are using Linux, it's recommended to run in the background:
```
nohup python -u sketch.py > log.txt &
```

## Reference
The codes of dataset preprocessing are based on  [PyRLA](https://github.com/wangshusen/PyRLA).

## Information
If you have any issues, please create a new issue of this repository or contact me via jinchengneng@gmail.com.