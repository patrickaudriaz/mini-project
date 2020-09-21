#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tabulate
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from database import transformToTextLabels
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.metrics import plot_confusion_matrix


def getMetricsTable(predictions, test_labels):
    """
    Generate a metrics table to evaluate predictions

    Parameters
    ----------

    predictions : array
        Predictions of a model
    test_labels : array
        Corresponding ground-truth

    Returns
    -------

    table : string
        Nicely formatted plain-text table with the computed metrics

    """

    # Compute metrics
    precision = precision_score(test_labels, predictions, average="micro")
    recall = recall_score(test_labels, predictions, average="micro")
    f1 = f1_score(test_labels, predictions, average="micro")
    accuracy = accuracy_score(test_labels, predictions)

    # Create table
    headers = ["Precision (avg)", "Recall (avg)", "F1 score (avg)", "Accuracy"]

    table = [[precision, recall, f1, accuracy]]

    return tabulate.tabulate(table, headers, tablefmt="rst", floatfmt=".3f")


def getTableHeader(model_name, model):
    """
    Generate a header for the metrics table

    Parameters
    ----------

    model_name : str
        Type of model (svm or rf)
    model : object
        Trained model from which to get the parameters

    Returns
    -------

    header : string
        Nicely formatted text header

    """

    header = f"Model used: {str(model_name)}\n" + "Parameters:\n"

    if model_name == "svm":
        header += (
            f"-kernel: {model.kernel}\n"
            + f"-gamma: {model.gamma}\n"
            + f"-C: {model.C}\n"
        )
    elif model_name == "rf":
        header += (
            f"-n_estimators: {model.n_estimators}\n"
            + f"-max_depth: {model.max_depth}\n"
            + f"-min_samples_split: {model.min_samples_split}\n"
            + f"-min_samples_leaf: {model.min_samples_leaf}\n"
            + f"-bootstrap: {model.bootstrap}\n"
        )

    return header


'''
def plot_confusion_matrix(y_pred, y_true,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    Plot the confusion matrix given predictions and ground-truth
    Source: https://scikit-learn.org/0.21/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    Parameters
    ----------

    y_pred : array
        Predictions of a model
    y_true : array
        Corresponding ground-truth
    normalize : bool
        Whether to normalize the matrix or not
    title : str
        Title of the graph
    cmap : LinearSegmentedColormap object
        Colour map of the confusion matrix

    Returns
    -------

    fig : Matplotlib figure
        Plot of the confusion matrix

    """
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return fig
'''


def evaluate(predictions, test_data, test_labels, output_dir, model_name, model):
    """
    Evaluate the predictions given the ground-truth. Save a table with
    the metrics and a png file with the confusion matrix.

    Parameters
    ----------

    predictions : array
        Predictions of a model
    test_labels : array
        Corresponding ground-truth
    output_dir : str
        Folder name in which to save table and figure
    model_name : str
        Model type (svm or rf)
    model : object
        trained model

    """

    logging.info(f"Starting evaluation...")

    # Get the metrics table
    table = getMetricsTable(predictions, test_labels)

    # Append header to the table
    table = getTableHeader(model_name, model) + table

    # Save the metrics table
    output_table = os.getcwd() + "/" + output_dir + "/table.rst"
    logging.info(f"Saving table at {output_table}")
    os.makedirs(os.path.dirname(output_table), exist_ok=True)
    with open(output_table, "wt") as f:
        f.write(table)

    # Numerical labels to text for the plot
    classes = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]

    fig, ax = plt.subplots(figsize=(17, 15))

    # Get the normalized confusion matrix
    fig = plot_confusion_matrix(
        model,
        test_data,
        test_labels,
        normalize="true",
        cmap=plt.cm.Blues,
        display_labels=classes,
        xticks_rotation="vertical",
        ax=ax,
    )

    fig.ax_.set(ylabel="True label", xlabel="Predicted label")

    title = f"Normalized confusion matrix\nModel: {model_name}"
    fig.ax_.set_title(title)

    # Save the confusion matrix
    output_figure = os.getcwd() + "/" + output_dir + "/confusion_matrix.png"
    logging.info(f"Saving confusion matrix at {output_figure}")
    plt.savefig(output_figure)
