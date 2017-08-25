import numpy as np
from sklearn import model_selection
from sklearn import datasets
from sklearn import svm

if __name__ == '__main__':
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
    print(X_train.shape, y_train.shape)
    clf = svm.SVC(kernel='linear', C=1)
    scores = model_selection.cross_val_score(clf, iris.data, iris.target, cv=5)
    print(scores)