from sklearn.svm import SVC

def train(X, Y, kernel, gamma, C):
    model = SVC(kernel, gamma, C)
    model.fit(X, Y)

    return model
