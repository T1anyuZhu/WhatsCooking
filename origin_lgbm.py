from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np
from origin_preprocess import load_data, SEED
from stacking_utils import get_base_predictions


def lgbm_pred():
    train_x, test_x, train_y, encoder, ids = load_data()

    # lgbm_clf = LGBMClassifier(n_estimators=100, min_child_weight=1, num_leaves=50,
    #                           colsample_bytree=0.45, seed=SEED)

    lgbm_clf = LGBMClassifier(n_estimators=100, seed=SEED)
    params = {
        'min_child_weight': np.arange(1, 11, 1),
        'num_leaves': np.arange(20, 101, 5),
        'colsample_bytree': np.arange(0.45, 1.01, 0.05),
    }
    clf = GridSearchCV(lgbm_clf, params, n_jobs=-1, verbose=1000)
    clf.fit(train_x, train_y)
    print(clf.cv_results_['mean_train_score'])
    print(clf.cv_results_['mean_test_score'])
    print(clf.best_params_)
    print(clf.best_score_)

    print(lgbm_clf.__class__.__name__)
    print(cross_val_score(lgbm_clf, train_x, train_y, n_jobs=-1, verbose=1000))
    # [0.7847395   0.78573045  0.78237247]

    base_train_x, base_test_x = get_base_predictions(lgbm_clf, train_x, test_x, train_y, kfold=5)
    np.save('origin/basemodels/lgbm_base_train_x', base_train_x)
    np.save('origin/basemodels/lgbm_base_test_x', base_test_x)



    # min_child_weight
    # 1 0.781917835772

    # colsample_bytree
    # 0.40 0.784155478453
    # 0.45 0.784281188716
    # 0.50 0.783300648665
