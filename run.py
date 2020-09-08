#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

import database
import algorithm
import evaluator


def main():

    # Load data and ground-truth
    train_data, train_labels, test_data, test_labels = database.load(
                                                        standardized=True,
                                                        printSize=True)
    train_labels = train_labels.ravel()

    # Training SVM model using radial kernel
    kernel = "rbf"
    gamma = 0.001
    C = 1000

    # Training
    model = algorithm.train(train_data, train_labels, kernel, gamma, C)

    predictions = algorithm.predict(test_data, model)

    # Evaluate the predictions
    evaluator.evaluate(predictions, test_labels)


if __name__ == "__main__":
    main()
