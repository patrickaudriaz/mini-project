#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import logging
import sys

import database
import algorithm
import argparse
import evaluator

logging.basicConfig(level=logging.INFO)


def main(args):
    if args.custom_train == True and args.custom_test == False:
        train_data, train_labels = database.load_custom_data(
            "Train", args.train_data, args.train_labels, printSize=True
        )

        test_data, test_labels = database.load_test(printSize=True)

    if args.custom_test == True and args.custom_train == False:
        train_data, train_labels = database.load_train(printSize=True)

        test_data, test_labels = database.load_custom_data(
            "Test", args.test_data, args.test_labels, printSize=True
        )

    if args.custom_test == True and args.custom_train == True:
        train_data, train_labels = database.load_custom_data(
            "Train", args.train_data, args.train_labels, printSize=True,
        )

        test_data, test_labels = database.load_custom_data(
            "Test", args.test_data, args.test_labels, printSize=True
        )

    if args.custom_train == False and args.custom_test == False:
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
        choices=["svm", "rf"],
    )
    parser.add_argument(
        "-gridsearch",
        type=str,
        default="n",
        help="Do Grid Search ? --> y or n [default]",
        dest="gridsearch",
        choices=["y", "n"],
    )
    parser.add_argument(
        "-output-folder",
        type=str,
        default="results",
        help="Path where to store the evaluation results (created if does not exist)",
        dest="output_folder",
    )

    parser.add_argument(
        "-custom-train",
        action="store_true",
        help="Flag to use custom data set for training. Need to specify -train-data (path to train_data.txt file) and -train-labels (path to train_labels.txt file)",
        dest="custom_train",
        default=False,
    )
    parser.add_argument(
        "-custom-test",
        action="store_true",
        help="Flag to use custom data set for prediction.Need to specify -test-data (path to test_data.txt file) and -test-labels (path to test_labels.txt file)",
        dest="custom_test",
        default=False,
    )

    args, rem_args = parser.parse_known_args(args)

    if args.custom_train:
        parser.add_argument(
            "-train-data",
            required=True,
            type=str,
            help="Path to train_data.txt file",
            dest="train_data",
        )
        parser.add_argument(
            "-train-labels",
            required=True,
            type=str,
            help="Path to train_labels.txt file",
            dest="train_labels",
        )

    if args.custom_test:
        parser.add_argument(
            "-test-data",
            required=True,
            type=str,
            help="Path to test_data.txt file",
            dest="test_data",
        )
        parser.add_argument(
            "-test-labels",
            required=True,
            type=str,
            help="Path to test_labels.txt file",
            dest="test_labels",
        )

    args = parser.parse_args(rem_args, namespace=args)

    return args


if __name__ == "__main__":
    args = get_args()
    logging.info(f"Using arguments: {args}")

    main(args)
