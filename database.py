#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
import requests, zipfile, io, os

logging.basicConfig(level=logging.INFO)

# URL and Root path of the original data folder
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATASET_PATH = "UCI HAR Dataset/"


def download_dataset():
    """
    Download raw dataset from url and unzip it
    """

    if not os.path.isdir("UCI HAR Dataset") or len(os.listdir("UCI HAR Dataset")) == 0:

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

    with open(DATASET_PATH + "activity_labels.txt", "r") as f:
        for row in f:
            num_label, text_label = row.replace("\n", "").split(" ")
            labels = np.where(labels == str(num_label), text_label, labels)

    return labels


def get_dataset_split(split="train"):
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
    with open(DATASET_PATH + split + "/X_" + split + ".txt", "r") as f:
        data = np.array([row.replace("  ", " ").strip().split(" ") for row in f])

    # Load labels
    with open(DATASET_PATH + split + "/y_" + split + ".txt", "r") as f:
        labels = np.array([row.replace("\n", "") for row in f], dtype=int)

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

    download_dataset()

    # Get training data
    train_data, train_labels = get_dataset_split("train")

    # Get testing data
    test_data, test_labels = get_dataset_split("test")

    logging.info(f"Dataset ready.")

    if printSize:
        logging.info(f"---Train samples: {train_data.shape[0]}")
        logging.info(f"---Test samples: {test_data.shape[0]}")

    # Standardization if required
    if standardized:
        from preprocessor import standardize

        train_data, test_data = standardize(train_data, test_data)

    return train_data, train_labels, test_data, test_labels


def load_train(standardized=False, printSize=False):
    """
    Get the train dataset and the corresponding labels

    Parameters
    ----------

    standardized : bool
        standardize the data before returning them or not

    Returns
    -------

    train_data : array
        all the data of the custom dataset
    train_labels: array
        all the corresponding labels


    """

    logging.info(f"Starting train dataset loading...")

    download_dataset()

    # Get training data
    train_data, train_labels = get_dataset_split("train")

    # Get testing data
    test_data, test_labels = get_dataset_split("test")

    logging.info(f"Train Dataset ready.")

    if printSize:
        logging.info(f"---Train samples: {train_data.shape[0]}")

    # Standardization if required
    if standardized:
        from preprocessor import standardize

        train_data, test_data = standardize(train_data, test_data)

    return train_data, train_labels


def load_test(standardized=False, printSize=False):
    """
    Get the test dataset and the corresponding labels

    Parameters
    ----------

    standardized : bool
        standardize the data before returning them or not

    Returns
    -------

    test_data : array
        all the test data of the dataset
    test_labels: array
        all the corresponding labels

    """

    logging.info(f"Starting test dataset loading...")

    download_dataset()

    # Get testing data
    test_data, test_labels = get_dataset_split("test")

    # Get training data
    train_data, train_labels = get_dataset_split("train")

    logging.info(f"Test Dataset ready.")

    if printSize:
        logging.info(f"---Test samples: {test_data.shape[0]}")

    # Standardization if required
    if standardized:
        from preprocessor import standardize

        train_data, test_data = standardize(train_data, test_data)

    return test_data, test_labels


def load_custom_data(type, data_path, labels_path, standardized=False, printSize=False):
    """
    Get the custom dataset and the corresponding labels
    from .txt files into a training and a testing set

    Parameters
    ----------

    type : string
        type of the data to be loaded (train or test)
    data_path : string
        path to the .txt file containing the data 
    data_labels : string
        path to the .txt file containing the labels
    standardized : bool
        standardize the data before returning them or not

    Returns
    -------

    data : array
        all the data of the custom dataset
    labels: array
        all the corresponding labels

    """
    logging.info(f"Using custom " + type + " Dataset...")

    # Load data
    with open(data_path, "r") as f:
        data = np.array([row.replace("  ", " ").strip().split(" ") for row in f])

    # Load labels
    with open(labels_path, "r") as f:
        labels = np.array([row.replace("\n", "") for row in f], dtype=int)

    logging.info(f"Custom " + type + " Dataset ready.")

    if printSize:
        logging.info(f"---" + type + " samples: " + str(data.shape[0]))

    return data, labels
