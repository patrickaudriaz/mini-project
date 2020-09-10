#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# Root path of the original data folder
DATASET_PATH = "dataset/"

def transformToTextLabels(labels):
    """
    Transform numerical labels to corresponding text

    Parameters
    ----------

    labels : array
        an array of numerical labels

    Returns
    -------

    labels : array
        same array with corresponding text labels

    """

    labels = labels.astype(str)
    
    with open(DATASET_PATH + "activity_labels.txt", 'r') as f:
        for row in f:
            num_label, text_label = row.replace('\n', '').split(' ')
            labels = np.where(labels==str(num_label), text_label, labels)
            
    return labels

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

    return data, labels


def load(standardized=False, printSize=False):
    """
    Get the dataset and the corresponding labels split
    into a training and a testing set

    Parameters
    ----------

    standardized : bool
        standardize the data before returning them or not

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

    # Print size if requested
    if printSize:
        logging.info(f"---Train samples: {train_data.shape[0]}")
        logging.info(f"---Test samples: {test_data.shape[0]}")

    # Standardization if required
    if standardized:
        from preprocessor import standardize

        train_data, test_data = standardize(train_data, test_data)

    return train_data, train_labels, test_data, test_labels