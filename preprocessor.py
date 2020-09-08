#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler
import logging

def standardize(train_data, test_data):
    """
    Standardize training and testing data

    Parameters
    ----------

    train_data : array
        Data on which to calculate the standardization parameters.
        The standardization is also applied on this subset.
    test_data : array
        Test subset on which to apply the standardization.

    Returns
    -------

    train_data : array
        Standardized training data
    test_data : array
        Standardized testing data

    """

    # Instantiate scikit standard scaler
    scaler = StandardScaler()
    
    # Fit and standardize on training data
    train_data_std = scaler.fit_transform(train_data)

    # Standardize testing data
    test_data_std = scaler.transform(test_data)

    logging.info(f"Dataset standardized.")

    return train_data_std, test_data_std