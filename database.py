#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# Root path of the original data folder
DATASET_PATH = "dataset/"

def getFeaturesNames():
    """
    TODO: useless?
    Get the features names of the dataset

    Returns
    -------

    array : features

    """

    with open(DATASET_PATH + "features.txt", 'r') as f:
        features = [row.replace('\n', '').split(' ')[1] for row in f]

    return features

def getDatasetSplit(split="train"):
    """
    Get data and ground-truth of selected split

    Parameters
    ----------

    split : str
        split (train or test) to return

    Returns
    -------

    data : array
        all the data of the split
    labels: array
        all the corresponding labels (ground-truth)

    """

    # Load data
    with open(DATASET_PATH + split + "/X_" + split + ".txt", 'r') as f:
        data = np.array([row.replace('  ', ' ').strip().split(' ') for row in f])

    # Load labels
    with open(DATASET_PATH + split + "/y_" + split + ".txt", 'r') as f:
        labels = np.array([row.replace('\n', '') for row in f], dtype=int)
        
        # Add column's label
        # labels = np.vstack(("Activity", labels))
    
    # Load features names
    # features = getFeaturesNames()

    # Stack columns names to the data
    # data = np.vstack((features, data))

    return data, labels


def load():
    """
    Get the dataset and the corresponding labels split
    into a training and a testing set

    Returns
    -------

    train_data : array
    train_labels: array
    test_data : array
    test_labels: array

    """
    logging.info(f"Starting dataset loading...")

    # Get training data
    train_data, train_labels = getDatasetSplit("train")

    # Get testing data
    test_data, test_labels = getDatasetSplit("test")

    logging.info(f"Dataset loaded.")

    return train_data, train_labels, test_data, test_labels