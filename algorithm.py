#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
import logging
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


logging.basicConfig(level=logging.INFO)


def train(X, Y, args):

    if args.model == "svm":
        logging.info(f"Training SVM model")

        if args.gridsearch == "n":
            logging.info(f"No grid search...")

            # Training SVM model using radial kernel and predefined parameters
            kernel = "rbf"
            gamma = 0.001
            C = 1000

            svm_model = SVC(kernel=kernel, gamma=gamma, C=C)
            svm_model.fit(X, Y)

            return svm_model

        else:
            logging.info(f"Doing grid search, it may take a while...")

            # Create the parameter grid
            params_grid = [
                {
                    "kernel": ["rbf"],
                    "gamma": [1e-1, 1e-2, 1e-3, 1e-4],
                    "C": [1, 10, 100, 1000],
                },
                {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
                {
                    "kernel": ["poly"],
                    "gamma": [1e-1, 1e-2, 1e-3, 1e-4],
                    "degree": [3, 4, 5, 6],
                    "C": [1, 10, 100, 1000],
                },
                {
                    "kernel": ["sigmoid"],
                    "gamma": [1e-1, 1e-2, 1e-3, 1e-4],
                    "C": [1, 10, 100, 1000],
                },
            ]

            svm_model = GridSearchCV(SVC(), params_grid, cv=3, verbose=10, n_jobs=-1)
            svm_model.fit(X, Y)

            print("Using hyperparameters --> \n", svm_model.best_params_)

            print()

            return svm_model

    else:
        logging.info(f"Training RF model")
        if args.gridsearch == "n":
            logging.info(f"No grid search...")

            # Training RF model using predefined parameters
            n_estimators = 100
            max_depth = 100
            min_samples_split = 2
            min_samples_leaf = 2
            bootstrap = True

            rf_model = RandomForestClassifier(
                max_depth=max_depth,
                n_estimators=n_estimators,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,
                random_state=42,
            )

            rf_model.fit(X, Y)

            return rf_model

        else:
            logging.info(f"Doing grid search, it may take a while...")

            n_estimators = [10, 25, 50, 100, 200]
            max_depth = [10, 25, 50, 75, 100]
            min_samples_split = [2, 5, 10, 15]
            min_samples_leaf = [1, 2, 5, 10]
            bootstrap = [True, False]

            param_grid = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "bootstrap": bootstrap,
            }

            rf = RandomForestRegressor(random_state=42)

            rf_model = GridSearchCV(
                estimator=rf, param_grid=param_grid, cv=3, verbose=10, n_jobs=-1
            )
            rf_model.fit(X, Y)

            print("Using hyperparameters --> \n", rf_model.best_params_)

            return rf_model


def predict(X, model):
    Y_pred = model.predict(X)

    return Y_pred
