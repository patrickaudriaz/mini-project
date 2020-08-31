import numpy as np
import pandas as pd
from sklearn.utils import shuffle

"""
PROTOCOLS = {
        'proto1': {'train': range(0, 30), 'test': range(30, 50)},
        'proto2': {'train': range(20, 50), 'test': range(0, 20)},
        }
"""


def genCSV():
    # turn txt files into usable csv
    return 0


def load():
    data = pd.read_csv("./data/data.csv")
    # train = pd.read_csv("./data/train.csv")
    # test = pd.read_csv("./data/test.csv")
    return data


def splitData(data, train_size=0.8):
    random_seed = 42
    train = shuffle(
        data.sample(frac=train_size, random_state=random_seed), random_state=random_seed
    )
    test = shuffle(data.drop(train.index), random_state=random_seed)

    return train, test


def getLabels(data):
    # Seperating Predictors and Outcome values from train and test sets
    Y_label = data.Activity.values.astype(object)
    return Y_label


def dropLabels(data):
    X = pd.DataFrame(data.drop(["Activity", "subject"], axis=1))
    return X

