#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import logging

import database
import algorithm
import argparse
import evaluator

logging.basicConfig(level=logging.INFO)


def main(args):

    # Load data and ground-truth
    train_data, train_labels, test_data, test_labels = database.load(
        standardized=True, printSize=True
    )
    train_labels = train_labels.ravel()

    # Training
    model = algorithm.train(train_data, train_labels, args)

    # Logging scores on train and test sets
    logging.info("---Training set accuracy: %f" % model.score(train_data, train_labels))
    logging.info("---Testing  set accuracy: %f" % model.score(test_data, test_labels))

    predictions = algorithm.predict(test_data, model)

    # Evaluate the predictions
    evaluator.evaluate(
        predictions, test_data, test_labels, args.output_folder, args.model, model
    )


def get_args(args=None):
    parser = argparse.ArgumentParser(
        """Train using a Support Vector Machine (SVM) model or a Random Forest 
        (RF) model. You can also train with (very slow) or without doing 
        Grid Search for performing hyper parameter tuning 
        
        Example: python run.py -model rf -gridsearch n -output-folder results
        """
    )
    parser.add_argument(
        "-model",
        type=str,
        default="svm",
        help="Use SVM or RF ? --> svm [default] or rf",
        dest="model",
    )
    parser.add_argument(
        "-gridsearch",
        type=str,
        default="n",
        help="Do Grid Search ? --> y or n [default]",
        dest="gridsearch",
    )
    parser.add_argument(
        "-output-folder",
        type=str,
        default="results",
        help="Path where to store the evaluation results (created if does not exist)",
        dest="output_folder",
    )

    args = parser.parse_args(args)

    return args


if __name__ == "__main__":
    args = get_args()
    logging.info(f"Using arguments: {args}")

    main(args)
