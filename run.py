import pandas as pd

import database
import preprocessor
import algorithm
import analysis


def main():
    data = database.load()
    train, test = database.splitData(data)

    # Frequency distribution of classes"
    train_outcome = pd.crosstab(index=train["Activity"], columns="count")
    print("\n")
    print(train_outcome)

    X_train = database.dropLabels(train)
    Y_train_label = database.getLabels(train)

    X_test = database.dropLabels(test)
    Y_test_label = database.getLabels(test)

    # Dimension of Train and Test set
    print("\nDimension of Train set", X_train.shape)
    print("Dimension of Test set", X_test.shape, "\n")

    num_cols = X_train._get_numeric_data().columns
    print("Number of numeric features:", num_cols.size)

    # Transforming non numerical labels into numerical labels
    Y_train, encoder_train = preprocessor.labelEncoder(Y_train_label)
    Y_test, encoder_test = preprocessor.labelEncoder(Y_test_label)

    # Standardize the Train and Test feature set
    X_train_scaled = preprocessor.standardizeTrain(X_train)
    X_test_scaled = preprocessor.standardizeTrain(X_test)

    # Training SVM model using radial kernel
    kernel = "rbf"
    gamma = 0.001
    C = 1000

    model = algorithm.train(X_train_scaled, Y_train, kernel, gamma, C)

    Y_pred = algorithm.predict(X_test_scaled, model)
    Y_pred_label = preprocessor.labelDecoder(Y_pred, encoder_test)

    analysis.getScore(model, X_train_scaled, Y_train, X_test_scaled, Y_test)

    # Confusion Matrix  and Accuracy Score
    analysis.confusionMatrix(Y_test_label, Y_pred_label)

    # Classification report
    analysis.classificationReport(Y_test_label, Y_pred_label)


if __name__ == "__main__":
    main()
