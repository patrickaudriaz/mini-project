import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# ### Load the Train and Test set
train = shuffle(pd.read_csv("./data/train.csv"))
test = shuffle(pd.read_csv("./data/test.csv"))


# Frequency distribution of classes"
train_outcome = pd.crosstab(index=train["Activity"], columns="count")
print("\n")
print(train_outcome)
print("\n")


# Visualizing Outcome Distribution
"""
temp = train["Activity"].value_counts()
df = pd.DataFrame({"labels": temp.index, "values": temp.values})

labels = df["labels"]
sizes = df["values"]

x_pos = [i for i, _ in enumerate(labels)]

plt.figure(1, [14, 6])
plt.bar(x_pos, sizes, width=0.6)
plt.xticks(x_pos, labels)
plt.show()
"""


# Normalize the Predictor(Feature Set) for SVM training
# Seperating Predictors and Outcome values from train and test sets
X_train = pd.DataFrame(train.drop(["Activity", "subject"], axis=1))
Y_train_label = train.Activity.values.astype(object)

X_test = pd.DataFrame(test.drop(["Activity", "subject"], axis=1))
Y_test_label = test.Activity.values.astype(object)

# Dimension of Train and Test set
print("Dimension of Train set", X_train.shape)
print("Dimension of Test set", X_test.shape, "\n")


# Transforming non numerical labels into numerical labels
encoder = preprocessing.LabelEncoder()

# encoding train labels
encoder.fit(Y_train_label)
Y_train = encoder.transform(Y_train_label)

# encoding test labels
encoder.fit(Y_test_label)
Y_test = encoder.transform(Y_test_label)

# Total Number of Continous and Categorical features in the training set
num_cols = X_train._get_numeric_data().columns
print("Number of numeric features:", num_cols.size)


names_of_predictors = list(X_train.columns.values)

# Scaling the Train and Test feature set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Training SVM model using radial kernel
final_model = SVC(kernel="rbf", gamma=0.001, C=1000)
final_model.fit(X_train_scaled, Y_train)


Y_pred = final_model.predict(X_test_scaled)
Y_pred_label = list(encoder.inverse_transform(Y_pred))

print("\n")
print("Training set score for SVM: %f" % final_model.score(X_train_scaled, Y_train))
print("Testing  set score for SVM: %f" % final_model.score(X_test_scaled, Y_test))
print("\n")

# ### Confusion Matrix  and Accuracy Score
print(confusion_matrix(Y_test_label, Y_pred_label))
print("\n")
print(classification_report(Y_test_label, Y_pred_label))
