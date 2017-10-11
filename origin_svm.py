import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

from origin_preprocess import load_data, SEED
from stacking_utils import get_base_predictions


def svm_pred():

    train_x, test_x, train_y, encoder, ids = load_data()
    svm_clf = LinearSVC(C=0.38, random_state=SEED)
    print(svm_clf.__class__.__name__)
    print(cross_val_score(svm_clf, train_x, train_y, n_jobs=-1, verbose=1000))
    # [0.79250547  0.78874727  0.79361606]

    base_train_x, base_test_x = get_base_predictions(svm_clf, train_x, test_x, train_y, kfold=5)
    np.save('origin/basemodels/svm_base_train_x', base_train_x)
    np.save('origin/basemodels/svm_base_test_x', base_test_x)

    # params = {
    #     # 'C': np.arange(0.3, 0.5, 0.02),
    #     'tol': np.arange(0.5, 1.6, 0.1) * 1e-4
    # }
    # clf = GridSearchCV(svm_clf, params, n_jobs=-1, verbose=1000)
    # clf.fit(train_x, train_y)
    # print(clf.cv_results_['mean_train_score'])
    # print(clf.cv_results_['mean_test_score'])
    # print(clf.best_params_)
    # print(clf.best_score_)

    # C
    # 0.38 0.791622668075
    # 0.40 0.791396389601
    # 0.50 0.790591843918
    # 0.60 0.790390707497
