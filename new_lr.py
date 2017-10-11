from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np

from new_preprocess import load_data, SEED

if __name__ == '__main__':
    train_x, test_x, train_y, encoder, ids = load_data()
    print(train_x.shape)
    encoder = LabelEncoder()
    lr_clf = LogisticRegression(random_state=SEED)
    # params = {
    #     # 'penalty': ['l1', 'l2'],
    #     # 'C': np.arange(2.11, 2.30, 0.01),
    #     'tol': np.arange(5, 16, 1) * 1e-5
    # }
    # clf = GridSearchCV(lr_clf, params, n_jobs=-1, verbose=1000)
    # clf.fit(train_x, train_y)
    # print(clf.cv_results_['mean_train_score'])
    # print(clf.cv_results_['mean_test_score'])
    # print(clf.best_params_)
    # print(clf.best_score_)
    # print(cross_val_score(lr_clf, train_x, train_y, n_jobs=-1, verbose=1000))
    # [0.7649099   0.76408477  0.76456384]

    # 0.77251471
    # 0.77947905667
    # 0.779931613617

