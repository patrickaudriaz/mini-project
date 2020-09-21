#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pytest
import database

import logging


"""Tests the database script"""

def test_downloadDataset():
    
    database.downloadDataset()

    assert os.path.exists("UCI HAR Dataset")
    assert os.path.isdir("UCI HAR Dataset")

    # Check if the right files and folders are present
    entries = os.listdir("UCI HAR Dataset")
    expected_files = ["activity_labels.txt", "features_info.txt", "features.txt",
                        "test", "train"]
    for f in expected_files:
        assert f in entries


def test_transformToTextLabels():

    num_labels = np.array([1, 2, 3, 4, 5, 6])
    labels = database.transformToTextLabels(num_labels)

    assert np.array_equal(labels, np.array(["WALKING", "WALKING_UPSTAIRS", 
                        "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]))


def test_getDatasetSplit():
    train_data, train_labels = database.getDatasetSplit("train")

    assert train_data.shape == (7352, 561)
    assert train_labels.shape == (7352, )
    assert min(train_labels) == 1
    assert max(train_labels) == 6

    test_data, test_labels = database.getDatasetSplit("test")

    assert test_data.shape == (2947, 561)
    assert test_labels.shape == (2947, )
    assert min(test_labels) == 1
    assert max(test_labels) == 6