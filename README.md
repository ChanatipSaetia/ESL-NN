# Encoded Sharing Layers Neural Network (ESL-NN)

## Overview
ESL-NN is a "Hierarchical Multi-Label Classification System" based on [PyTorch](http://pytorch.org/) which is free for download and free for non-commecial use.

## Features
* Multi-class classification
* Multi-label classification
* Hierarchical multi-label classification with tree and DAG structure.

## Supported Platforms
* Linux

## Team Members

1. Peerapon Vateekul (Supervisor)
2. Chanatip Saetia

# Folder structure
- ``/assemble_classifier`` - Main classifiers code which store ESLNN, SHLNN, HMCLMLP
- ``/classifier`` - Classifiers of each level code which will be composed together in main classifiers in ``/assemble_classifier``
- ``/data`` - Data preprocessing code and Dataset class which is specifically used for our main classifier
- ``/embedding`` - Document embedding code
- ``/evaluation`` - Evaluation metric code
- ``Train.ipynb`` - A demo jupyter notebook for a training process
- ``Test.ipynb`` - A demo jupyter notebook for a testing process