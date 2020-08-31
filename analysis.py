#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import confusion_matrix, classification_report

def getScore(model, X_train, Y_train, X_test, Y_test):
    print("\n")
    print("Training set score for SVM: %f" % model.score(X_train, Y_train))
    print("Testing  set score for SVM: %f" % model.score(X_test, Y_test))
    print("\n")


def classificationReport(Y_test_label, Y_pred_label):
    print(confusion_matrix(Y_test_label, Y_pred_label))
    print("\n")


def confusionMatrix(Y_test_label, Y_pred_label):
    print(classification_report(Y_test_label, Y_pred_label))
