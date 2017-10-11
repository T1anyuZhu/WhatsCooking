import numpy as np
from sklearn.model_selection import StratifiedKFold

from origin_classifier import SEED


def get_base_predictions(clf, train_x, test_x, train_y, kfold=5):
    num_class = np.max(train_y) + 1
    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=SEED)
    base_train_x = np.zeros((train_x.shape[0], num_class))
    base_test_x = np.zeros((kfold, test_x.shape[0], num_class))
    for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
        model_train_x = train_x[train_index, :]
        model_train_y = train_y[train_index]
        model_test_x = train_x[test_index, :]
        clf.fit(model_train_x, model_train_y)
        try:
            base_train_x[test_index, :] = clf.predict_proba(model_test_x)
            base_test_x[i, :, :] = clf.predict_proba(test_x)
        except:
            def to_categorical(y):
                n = y.shape[0]
                num_class = np.max(y) + 1
                categorical = np.zeros((n, num_class))
                categorical[np.arange(n), y] = 1
                return categorical

            base_train_x[test_index, :] = to_categorical(clf.predict(model_test_x))
            base_test_x[i, :, :] = to_categorical(clf.predict(test_x))
    return base_train_x, base_test_x.mean(axis=0)
