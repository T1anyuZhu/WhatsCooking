import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
from origin_preprocess import load_data, SEED
from stacking_utils import get_base_predictions


def sgd_pred():

    train_x, test_x, train_y, encoder, ids = load_data()
    sgd_clf = SGDClassifier(alpha=4e-5, random_state=SEED, n_jobs=-1)
    print(sgd_clf.__class__.__name__)
    print(cross_val_score(sgd_clf, train_x, train_y, n_jobs=-1, verbose=1000))
    # [0.77870768  0.77796214  0.78056142]

    base_train_x, base_test_x = get_base_predictions(sgd_clf, train_x, test_x, train_y, kfold=5)
    np.save('origin/basemodels/sgd_base_train_x', base_train_x)
    np.save('origin/basemodels/sgd_base_test_x', base_test_x)

    # print(cross_val_score(sgd_clf, train_x, train_y, n_jobs=-1))
    # params = {
    # 'alpha': np.arange(4, 5, 1) * 1e-5,
    #     'n_iter': np.arange(1, 10, 1)
    # }
    # clf = GridSearchCV(sgd_clf, params, n_jobs=-1, verbose=1000)
    # clf.fit(train_x, train_y)
    # print(clf.cv_results_['mean_train_score'])
    # print(clf.cv_results_['mean_test_score'])
    # print(clf.best_params_)
    # print(clf.best_score_)

    # alpha
    # 3e-5 0.775632322623
    # 5e-5 0.779076783829

if __name__ == '__main__':
    sgd_pred()