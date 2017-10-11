import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

from origin_preprocess import load_data, SEED
from stacking_utils import get_base_predictions


def rf_pred():

    train_x, test_x, train_y, encoder, ids = load_data()
    rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    print(rf_clf.__class__.__name__)
    print(cross_val_score(rf_clf, train_x, train_y, n_jobs=-1, verbose=1000))
    # [0.75239388  0.74334414  0.74622698]

    base_train_x, base_test_x = get_base_predictions(rf_clf, train_x, test_x, train_y, kfold=5)
    np.save('origin/basemodels/rf_base_train_x', base_train_x)
    np.save('origin/basemodels/rf_base_test_x', base_test_x)


if __name__ == '__main__':
    rf_pred()
