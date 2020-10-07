#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
import requests, zipfile, io, os

logging.basicConfig(level=logging.INFO)

# URL and Root path of the original data folder
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATASET_PATH = "UCI HAR Dataset"
TRAIN_DATA = DATASET_PATH + "/train/X_train.txt"
TRAIN_LABELS = DATASET_PATH + "/train/y_train.txt"
TEST_DATA = DATASET_PATH + "/test/X_test.txt"
TEST_LABELS = DATASET_PATH + "/test/y_test.txt"


def download_dataset():
    """
    Download raw dataset from url and unzip it
    """

    if not os.path.isdir(DATASET_PATH) or len(os.listdir(DATASET_PATH)) == 0:

        logging.info(f"Dataset not locally available, downloading...")

        r = requests.get(URL)

        if not r.ok:
            logging.info(f"Error while downloading: {r.status_code}")
            return

        logging.info(f"Extracting raw data...")

        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
    else:
        logging.info(f"Dataset already available, skipping download.")


def transform_to_text_labels(labels):
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

    with open(DATASET_PATH + "/activity_labels.txt", "r") as f:
        for row in f:
            num_label, text_label = row.replace("\n", "").split(" ")
            labels = np.where(labels == str(num_label), text_label, labels)

    return labels


def get_dataset_split(data_path, labels_path):
    """
    Get data and ground-truth of selected split

    Parameters
    ----------

    data_path : str
        data file location
    labels_path : str
        labels file location

    Returns
    -------

    data : array
        all the data of the split
    labels: array
        all the corresponding labels (ground-truth)

    """

    # Load data
    with open(data_path, "r") as f:
        data = np.array([row.replace("  ", " ").strip().split(" ") for row in f])

    # Load labels
    with open(labels_path, "r") as f:
        labels = np.array([row.replace("\n", "") for row in f], dtype=int)

    return data, labels


def load(
    standardized=False,
    printSize=False,
    train_data_path=None,
    train_labels_path=None,
    test_data_path=None,
    test_labels_path=None,
):
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

    download_dataset()

    # Get training data
    if train_data_path and train_labels_path:
        train_data, train_labels = get_dataset_split(train_data_path, train_labels_path)
        logging.info(f"Custom Train Dataset loaded.")
    else:
        train_data, train_labels = get_dataset_split(TRAIN_DATA, TRAIN_LABELS)

    # Get testing data
    if test_data_path and test_labels_path:
        test_data, test_labels = get_dataset_split(test_data_path, test_labels_path)
        logging.info(f"Custom Test Dataset loaded.")
    else:
        test_data, test_labels = get_dataset_split(TEST_DATA, TEST_LABELS)

    logging.info(f"Dataset ready.")

    if printSize:
        logging.info(f"---Train samples: {train_data.shape[0]}")
        logging.info(f"---Test samples: {test_data.shape[0]}")

    # Standardization if required
    if standardized:
        from .preprocessor import standardize

        train_data, test_data = standardize(train_data, test_data)

    return train_data, train_labels, test_data, test_labels
