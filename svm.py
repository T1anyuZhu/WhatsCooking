from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np
from sklearn.svm import LinearSVC

from preprocess import load_data

SEED = 2017

if __name__ == '__main__':
    train_df, test_df, trainX, testX, trainY, encoder = load_data()
    svm_clf = LinearSVC(random_state=SEED)
    score = cross_val_score(svm_clf, trainX, trainY)
    print(score)
