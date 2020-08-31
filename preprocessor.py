#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def labelEncoder(data_label):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(data_label)
    Y = encoder.transform(data_label)

    return Y, encoder


def labelDecoder(data, encoder):
    Y_pred_label = list(encoder.inverse_transform(data))

    return Y_pred_label


def standardizeTrain(data):
    scaler = StandardScaler()
    # analyze method
    X_train_scaled = scaler.fit_transform(data)

    return X_train_scaled


def standardizeTest(data):
    scaler = StandardScaler()
    # analyze method
    X_test_scaled = scaler.transform(data)

    return X_test_scaled
