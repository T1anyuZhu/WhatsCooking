from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np
from origin_preprocess import load_data, SEED
from stacking_utils import get_base_predictions


def lr_pred():

    train_x, test_x, train_y, encoder, ids = load_data()
    lr_clf = LogisticRegression(penalty='l1', C=3.02, random_state=SEED, n_jobs=-1)
    print(lr_clf.__class__.__name__)
    print(cross_val_score(lr_clf, train_x, train_y, n_jobs=-1, verbose=1000))
    # [0.78850939  0.78512708  0.79188047]

    base_train_x, base_test_x = get_base_predictions(lr_clf, train_x, test_x, train_y, kfold=5)
    np.save('origin/basemodels/lr_base_train_x', base_train_x)
    np.save('origin/basemodels/lr_base_test_x', base_test_x)


    # params = {
    # 'penalty': ['l1'],
    # 'C': np.arange(3.0, 3.1, 0.02),
    # 'tol': np.arange(5, 15, 2) * 1e-5,
    # }
    # 0.788505053553
    # clf = GridSearchCV(lr_clf, params, n_jobs=-1, verbose=1000)
    # clf.fit(train_x, train_y)
    # print(clf.cv_results_['mean_train_score'])
    # print(clf.cv_results_['mean_test_score'])
    # print(clf.best_params_)
    # print(clf.best_score_)
    # print(lr_clf.__class__.__name__)

