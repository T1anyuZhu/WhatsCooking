import numpy as np

from sklearn.model_selection import StratifiedKFold

SEED = 2017


def get_y_pred(y_pred, nlabel):
    n = len(y_pred)
    rt = np.zeros((n, nlabel))
    rt[range(0, n), y_pred] = 1
    return rt


def get_oof(clf, trainX, trainY, testX, kfold=5):
    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=SEED)
    ntrain, ntest, nlabel = trainX.shape[0], testX.shape[0], len(set(trainY))
    rt_trainX, rt_testX = np.zeros((ntrain, nlabel)), np.zeros((kfold, ntest, nlabel))
    for i, (train_index, test_index) in enumerate(kf.split(trainX, trainY)):
        training_samples = trainX[train_index]
        training_samples_labels = trainY[train_index]
        test_samples = trainX[test_index]
        clf.fit(training_samples, training_samples_labels)
        training_y_pred = clf.predict(test_samples)
        rt_trainX[test_index, :] = get_y_pred(training_y_pred, nlabel)
        testing_y_pred = clf.predict(testX)
        rt_testX[i, :] = get_y_pred(testing_y_pred, nlabel)
    return rt_trainX, np.mean(rt_testX, axis=0)
