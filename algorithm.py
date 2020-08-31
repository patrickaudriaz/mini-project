from sklearn.svm import SVC


def train(X, Y, kernel, gamma, C):
    model = SVC(kernel=kernel, gamma=gamma, C=C)
    model.fit(X, Y)

    return model


def predict(X, model):
    Y_pred = model.predict(X)

    return Y_pred
