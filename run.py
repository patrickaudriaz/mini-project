#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

import database
import preprocessor
import algorithm
import analysis


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

    # Get score
    analysis.getScore(model, train_data, train_labels, test_data, test_labels)

    # Confusion Matrix  and Accuracy Score
    analysis.confusionMatrix(test_labels, predictions)

    # Classification report
    analysis.classificationReport(test_labels, predictions)


if __name__ == "__main__":
    main()
