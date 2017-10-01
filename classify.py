import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from preprocess import load_data
from stackinghelper import StackingHelper, get_oof

SEED = 2017  # for reproducibility

if __name__ == '__main__':
    train_df, test_df, trainX, testX, trainY, encoder = load_data()

    if not os.path.exists('stacking/lr_second_layer_trainX.npy'):
        lr_clf = StackingHelper(LogisticRegression(C=5.0, n_jobs=-1, random_state=SEED))
        lr_second_layer_trainX, lr_second_layer_testX = get_oof(lr_clf, trainX, trainY, testX)
        np.save('stacking/lr_second_layer_trainX', lr_second_layer_trainX)
        np.save('stacking/lr_second_layer_testX', lr_second_layer_testX)
        # [ 0.78571967  0.78452372  0.78531542]
    else:
        lr_second_layer_trainX = np.load('stacking/lr_second_layer_trainX.npy')
        lr_second_layer_testX = np.load('stacking/lr_second_layer_testX.npy')

    if not os.path.exists('stacking/lgbm_second_layer_trainX.npy'):
        lgbm_clf = StackingHelper(LGBMClassifier(n_estimators=1500, colsample_bytree=0.5, learning_rate=0.1, seed=SEED))
        lgbm_second_layer_trainX, lgbm_second_layer_testX = get_oof(lgbm_clf, trainX, trainY, testX)
        np.save('stacking/lgbm_second_layer_trainX', lgbm_second_layer_trainX)
        np.save('stacking/lgbm_second_layer_testX', lgbm_second_layer_testX)
        # [ 0.77712433  0.7778113   0.77550558]
    else:
        lgbm_second_layer_trainX = np.load('stacking/lgbm_second_layer_trainX.npy')
        lgbm_second_layer_testX = np.load('stacking/lgbm_second_layer_testX.npy')

    if not os.path.exists('stacking/sgd_second_layer_trainX.npy'):
        sgd_clf = StackingHelper(SGDClassifier(n_iter=17, n_jobs=-1, random_state=SEED))
        sgd_second_layer_trainX, sgd_second_layer_testX = get_oof(sgd_clf, trainX, trainY, testX)
        np.save('stacking/sgd_second_layer_trainX', sgd_second_layer_trainX)
        np.save('stacking/sgd_second_layer_testX', sgd_second_layer_testX)
        # [0.77079092  0.77042009  0.77195895]
    else:
        sgd_second_layer_trainX = np.load('stacking/sgd_second_layer_trainX.npy')
        sgd_second_layer_testX = np.load('stacking/sgd_second_layer_testX.npy')

    if not os.path.exists('stacking/knn_second_layer_trainX.npy'):
        knn_clf = StackingHelper(KNeighborsClassifier(n_jobs=-1))
        knn_second_layer_trainX, knn_second_layer_testX = get_oof(knn_clf, trainX, trainY, testX)
        np.save('stacking/knn_second_layer_trainX', knn_second_layer_trainX)
        np.save('stacking/knn_second_layer_testX', knn_second_layer_testX)
        # [0.73143331  0.72667622  0.73302143]
    else:
        knn_second_layer_trainX = np.load('stacking/knn_second_layer_trainX.npy')
        knn_second_layer_testX = np.load('stacking/knn_second_layer_testX.npy')

    if not os.path.exists('stacking/rf_second_layer_trainX.npy'):
        rf_clf = StackingHelper(RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=SEED))
        rf_second_layer_trainX, rf_second_layer_testX = get_oof(rf_clf, trainX, trainY, testX)
        np.save('stacking/rf_second_layer_trainX', rf_second_layer_trainX)
        np.save('stacking/rf_second_layer_testX', rf_second_layer_testX)
        # [0.75103672  0.74522966  0.74660429]
    else:
        rf_second_layer_trainX = np.load('stacking/rf_second_layer_trainX.npy')
        rf_second_layer_testX = np.load('stacking/rf_second_layer_testX.npy')

    if not os.path.exists('stacking/svm_second_layer_trainX.npy'):
        svm_clf = StackingHelper(LinearSVC(random_state=SEED))
        svm_second_layer_trainX, svm_second_layer_testX = get_oof(svm_clf, trainX, trainY, testX)
        np.save('stacking/svm_second_layer_trainX', svm_second_layer_trainX)
        np.save('stacking/svm_second_layer_testX', svm_second_layer_testX)
        # [0.78406092  0.7815069   0.78569273]
    else:
        svm_second_layer_trainX = np.load('stacking/svm_second_layer_trainX.npy')
        svm_second_layer_testX = np.load('stacking/svm_second_layer_testX.npy')

    if not os.path.exists('stacking/xgb_second_layer_trainX.npy'):
        xgb_clf = StackingHelper(XGBClassifier(n_estimators=2000, seed=SEED))
        xgb_second_layer_trainX, xgb_second_layer_testX = get_oof(xgb_clf, trainX, trainY, testX)
        np.save('stacking/xgb_second_layer_trainX.npy', xgb_second_layer_trainX)
        np.save('stacking/xgb_second_layer_testX', xgb_second_layer_testX)
    else:
        xgb_second_layer_trainX = np.load('stacking/xgb_second_layer_trainX.npy')
        xgb_second_layer_testX = np.load('stacking/xgb_second_layer_testX.npy')

    second_layer_trainX = np.hstack(
        (lr_second_layer_trainX, lgbm_second_layer_trainX, sgd_second_layer_trainX, knn_second_layer_trainX,
         rf_second_layer_trainX, svm_second_layer_trainX, xgb_second_layer_trainX))
    second_layer_trainY = trainY
    second_layer_testX = np.hstack(
        (lr_second_layer_testX, lgbm_second_layer_testX, sgd_second_layer_testX, knn_second_layer_testX,
         rf_second_layer_testX, svm_second_layer_testX, xgb_second_layer_testX))

    second_layer_lgbm = LGBMClassifier(learning_rate=0.019849950000000002, n_estimators=100,
                                       min_child_samples=13, min_child_weight=1, colsample_bytree=0.55,
                                       reg_alpha=0.0018367346938775511, reg_lambda=0.098673469387755106, seed=SEED)

    # second_layer_lgbm = LGBMClassifier(num_leaves=19, learning_rate=0.16510204081632654, n_estimators=100,
    #                                    min_child_samples=13,
    #                                    reg_alpha=0.0018367346938775511, reg_lambda=0.098673469387755106, seed=SEED)

    # second_layer_lgbm.fit(second_layer_trainX, second_layer_trainY)
    # print(cross_val_score(second_layer_lgbm, second_layer_trainX, second_layer_trainY))
    # 0.807738723789
    # 0.807763865842
    # 0.807763865842
    # 0.807839292
    # 0.807889576105
    # 0.807965002263
    # 0.807990144315
    # 0.808065570473
    # 0.808719263841
    # learning_rate = 0.16510204081632654, n_estimators = 100,
    # min_child_samples = 13,
    # reg_alpha = 0.0018367346938775511, reg_lambda = 0.098673469387755106, seed = SEED

    # 0.807361593
    # 0.807613013526 'learning_rate': 0.01985
    # 0.807487303263 'learning_rate': 0.019849473684210524
    # 0.807638155579 'learning_rate': 0.019849947368421054
    # 0.807663297632 'learning_rate': 0.019849952631578946
    # 0.808367275104 'max_bin': 200
    # 0.808593553578 'subsample': 1.0000000000000004
    # 0.808819832051 'reg_alpha': 0.14285714285714285
    # 0.808920400261 'reg_alpha': 0.14591836734693878
    params = {
        # 'num_leaves': np.arange(3, 6)
        # 'learning_rate': np.linspace(0.01984995, 0.01984996, num=20),
        # 'colsample_bytree': np.arange(0.45, 0.55, 0.02),
        # 'min_child_weight': np.arange(1, 11, 2),
        # 'reg_alpha': np.linspace(0, 0.01),
        # 'subsample_freq': np.arange(1, 6),
        'reg_alpha': np.linspace(0.14, 0.15)
        # 'reg_lambda': np.linspace(0.095, 0.105),
        # 'colsample_bytree': np.linspace(0.6, 1.0),
        # 'min_child_samples': np.linspace(5, 15, dtype=int),
    }

    clf = GridSearchCV(second_layer_lgbm, params)
    clf.fit(second_layer_trainX, second_layer_trainY)
    # score = cross_val_score(second_layer_lgbm, second_layer_trainX, second_layer_trainY)
    # print(score)
    print(clf.best_score_)
    print(clf.best_params_)
    #
    # y_pred = second_layer_lgbm.predict(second_layer_testX)
    # y_pred = encoder.inverse_transform(y_pred)
    #
    # sub = pd.DataFrame({
    #     'id': test_df['id'],
    #     'cuisine': y_pred
    # }, columns=['id', 'cuisine'])
    # sub.to_csv('output/stacking_submission2.csv', index=False)
