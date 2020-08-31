import pandas as pd

import database
import preprocessor
import algorithm
import analysis


def main():

    # TODO: Required? Need a refactor
    # Frequency distribution of classes"
    # train_outcome = pd.crosstab(index=train["Activity"], columns="count")
    # print("\n")
    # print(train_outcome)

    # Load data and ground-truth
    train_data, train_labels, test_data, test_labels = database.load()
    train_labels = train_labels.ravel()

    # Standardize training and testing set
    # TODO: verify what is required
    train_data = preprocessor.standardizeTrain(train_data)
    test_data = preprocessor.standardizeTrain(test_data)

    # Dimension of Train and Test set
    print("\nDimension of training set", train_data.shape)
    print("Dimension of testing set", test_data.shape, "\n")

    # TODO: useful? I do not think so
    num_cols = train_data.shape[1]
    print("Number of numeric features:", num_cols)

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
