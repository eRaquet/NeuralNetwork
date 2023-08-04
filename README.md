# NeuralNetwork

## Description

This repository contains a simple, python based, neural network module.  It is built to be easially adaptable/customizable for whatever purpose you have for it.  I was largely inspired to make this project by a video by [Sebastian League](https://www.youtube.com/watch?v=hfMk-kjRv4c&t=2232s).  There are a few (or many!!!) improvements I plan to make in the future.

## Structure

[neuralNetwork.py](.\neuralNetwork.py) is the main module script.  The training cost functions (used for evaluating the quality of the network) are supplied by [costFunctions.py](.\costFunctions.py).  Each layer of the network can be activated by a linear or non-linear activation function supplied by [activationFunctions.py](.\activationFunctions.py).  Both activationFunctions.py and costFunctions.py can easially be suplimented with custom activation and cost functions.  [createData.py](.\creatData.py) is a data simulator, allowing you to create custom functions, add noise, and then save data for the network to train on.  [moduleTest.py](.\moduleTest.py) demonstrates some of the capabilities of the neuralNetwork module.

## Network Storage

This module can save neural networks using [HDF5](https://docs.hdfgroup.org/hdf5/develop/_h5_d__u_g.html) datasets.  It also uses this data format to store the training/test data, but that could be changed for the convinience of the user.  Stored networks are stored in the [savedNetworks](.\savedNetworks) folder and can be loaded into a program, trained or used, and then saved again.  Datasets are stored similarly in the [trainingDatasets](.\trainingDatasets) folder.

##