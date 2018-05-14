# Encoded Sharing Layers Neural Network (ESL-NN)

## Overview
ESL-NN is a "Hierarchical Text Categorization System" based on [PyTorch](http://pytorch.org/) which is free for download and free for non-commecial use.

## Features
* Multi-class classification
* Multi-label classification
* Hierarchical multi-label classification with tree and DAG structure.

## Supported Platforms
* MacOS
* Linux

## Publications
**Official ESL-NN algorithm:**<br>
Peerapon Vateekul and Chanatip Saetia, “Enhance Accuracy of Hierarchical Text Categorization Based on Deep Learning Network Using Embedding Strategies,” Computer Science and Software Engineering (JCSSE), 2018 15th International Joint Conference on. IEEE, 2018. (In review process).

## Team Members

1. Peerapon Vateekul (Supervisor)
2. Chanatip Saetia

## Related Works
* [HR-SVM Project](https://sites.google.com/site/hrsvmproject/) - a "Hierarchical Multi-Label Classification System" based on LIBSVM and HEMKit
* Mongkud Klungpornkun and Peerapon Vateekul, “Hierarchical Text Categorization Using Level Based Neural Networks of Word Embedding Sequences with Sharing Layer Information,” Computer Science and Software Engineering (JCSSE), 2017 14th International Joint Conference on. IEEE, 2017.

# Tutorial
## Training Process
1. Store dataset and hierarchical structure(if the task is hierarchical classification) in the /data in the following structure. A format in dataset and hierarchical file is described in Data format section
~~~~
data/
    <your_dataset_name>/
        <your_train_file_name>.txt
        <your_test_file_name>.txt
        hierarchy.txt
~~~~
Example
~~~~
data/
    wipo_d/
        train.txt
        test.txt
        hierarchy.txt
~~~~
2. Config the classifier by edit `config.json`. The configuration setting is explained in Configuration section
3. Open a terminal and run
~~~~
./classifier/Predictor
~~~~
4. After the system is finished, the all result file will store in the `export/<your_dataset_name>` folder. In the folder will store this following file.
    * `result.txt` - store the evaluation result in term of f1 macro and micro
    * `prediction` - a folder which store the prediction of each instance in a dataset
    * `probability_prediction` - a folder which store the probability prediction of each instance in a dataset
    * `doc2vec` - a folder which store the document embedding of each instance in a dataset
    * `level_i.model` - the model of ith level classifier
    * `doc2vec.model` - the model of document embedding

## Evaluation Process
1. Store a dataset similar with training process
2. Config the classifier by edit `evaluater_config.json`. The configuration setting is explained in Configuration section
3. Open a terminal and run
~~~~
./classifier/Evaluater
~~~~
4. After the system is finished, the evaluation result file will store in the `export/<your_train_model_folder>` folder. In the folder will store this following file.
    * `result_<your_file_name>.txt` - store the evaluation result in term of f1 macro and micro
    * `prediction/<your_file_name>.txt` - the prediction of each instance in a dataset
    * `probability_prediction/<your_file_name>.txt` - the probability prediction of each instance in a dataset

# Configuration
## Training Configuration
the configuration for training process which determined in `"config.json"`
### Parameter

| Parameter| Description |
|-----------------|:-----------|
| **data_name**| a name of dataset which is same as folder name in ``/data`` |
| **train_file_name**       | a name of a file that store train dataset |
| **test_file_name**        | a name of a file that store test dataset |
| **classification_type**   | a type of classification <ul><li>"multi-class" : Multi-class classification</li><li>"multi-label" : Multi-label classification</li><li>"hierarchical" : Hierarchical multi-label classification</li></ul> |
| **test_split**            | Enable/disable split some part of train data to be test data <br><sub>\*In this case the **test_file_name** data isn't used</sub> |
| **predict_test**          | Enable/disable prediction test data in evaluation process |
| **evaluate_test**         | Enable/disable evaluation test data in evaluation process |
| **correction**            | Enable/disable label correction after prediction in evaluation process <br><sub>\*This parameter is used when **classification_type** is "hierarchical" only</sub>|
| **mandatory_leaf**        | Select the classification task is mandatory leaf or not <br><sub>\*This parameter is used when **classification_type** is "hierarchical" only</sub> |
| **hidden**                | a list of the number of hidden nodes in each level. <br><sub>\*This can be set to "auto" where the system will calculate the number of hidden nodes automatically</sub> |
| **target_hidden**         | a list of the number of hidden nodes in each shared layer. <br><sub>\*This can be set to "auto" where the system will calculate the number of hidden nodes automatically</sub> |
| **embedding_size**        | a size of document embedding |

### Example
~~~~
{
    "data_name": "wipo_d",
    "train_file_name": "data.txt",
    "test_file_name": "test.txt",
    "classification_type": "hierarchical",
    "test_split": false,
    "predict_test": false,
    "evaluate_test": false,
    "correction": true,
    "mandatory_leaf": false,
    "hidden": [100,200,300],
    "target_hidden": "auto",
    "embedding_size": 150,
    "embedding_type": "OPD"
}
~~~~

## Evaluating Configuration
the configuration for evaluating process which determined in `"evaluater_config.json"`
### Parameter

| Parameter | Description |
|-----------------|:-----------|
| **train_model_folder** | a name of folder in `/export` where the train model is stored |
| **file_name** | a name of a file that store dataset that will be used to predict and evaluate |

### Example
~~~~
{
    "train_model_folder": "wipo_d",
    "file_name": "test.txt"
}
~~~~

# Data file format
The data must be tokenized and store sequences of words in documents to a file with this following format.
~~~~
sport,football:[football,players,are,tried]
~~~~
This is an example format of one document which `sport` and `football` is its labels and `[football,players,are,tried]` is the sequence of words in that document

# Hierarchy file format
The hierarchy can be explain in this following format
~~~~
hobby sport
sport football
sport basketball
~~~~
This is an example format of the hierarchy file. Two categories store in one line where the first categories is a parent node of the second categories. For instance, `hobby` is a parent node of `sport`