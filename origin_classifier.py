import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

import stackinghelper
from origin_preprocess import load_data

SEED = 2017

if __name__ == '__main__':
    train_df, test_df, trainX, testX, trainY, encoder = load_data()
    if not os.path.exists('basemodels/new_base_trainX.npy') or not os.path.exists('basemodels/new_base_testX.npy'):
        if not os.path.exists('basemodels/lr_base_trainX.npy') or not os.path.exists(
                'basemodels/lr_base_testX.npy'):
            lr_clf = LogisticRegression(C=2.92, tol=7e-5, penalty='l1', n_jobs=-1, random_state=SEED)
            # print(cross_val_score(lr_clf, trainX, trainY))
            # [0.788057    0.78459914  0.79112587]
            lr_base_trainX, lr_base_testX = stackinghelper.get_oof(lr_clf, trainX, trainY, testX)
            np.save('basemodels/lr_base_trainX', lr_base_trainX)
            np.save('basemodels/lr_base_testX', lr_base_testX)
        else:
            lr_base_trainX = np.load('basemodels/lr_base_trainX.npy')
            lr_base_testX = np.load('basemodels/lr_base_testX.npy')

        if not os.path.exists('basemodels/sgd_base_trainX.npy') or os.path.exists('basemodels/sgd_base_testX.npy'):
            sgd_clf = SGDClassifier(n_iter=30, alpha=3e-5, n_jobs=-1, random_state=SEED)
            # print(cross_val_score(sgd_clf, trainX, trainY))
            # [ 0.78873558  0.78392036  0.7902958 ]
            sgd_base_trainX, sgd_base_testX = stackinghelper.get_oof(sgd_clf, trainX, trainY, testX)
            np.save('basemodels/sgd_base_trainX', sgd_base_trainX)
            np.save('basemodels/sgd_base_testX', sgd_base_testX)
        else:
            sgd_base_trainX = np.load('basemodels/sgd_base_trainX.npy')
            sgd_base_testX = np.load('basemodels/sgd_base_testX.npy')

        if not os.path.exists('basemodels/svm_base_trainX.npy') or not os.path.exists(
                'basemodels/svm_base_testX.npy'):
            svm_clf = LinearSVC(C=0.3341, random_state=SEED)
            # print(cross_val_score(svm_clf, trainX, trainY))
            # [ 0.7914499   0.78625839  0.79233323]
            svm_base_trainX, svm_base_testX = stackinghelper.get_oof(svm_clf, trainX, trainY, testX)
            np.save('basemodels/svm_base_trainX', svm_base_trainX)
            np.save('basemodels/svm_base_testX', svm_base_testX)
        else:
            svm_base_trainX = np.load('basemodels/sgd_base_trainX.npy')
            svm_base_testX = np.load('basemodels/sgd_base_testX.npy')

        if not os.path.exists('basemodels/lgbm_base_trainX.npy') or not os.path.exists(
                'basemodels/lgbm_base_testX.npy'):
            lgbm_clf = LGBMClassifier(learning_rate=0.140, colsample_bytree=0.39, min_child_weight=1,
                                      reg_alpha=0.0071,
                                      n_estimators=100,
                                      num_leaves=49,
                                      seed=SEED)
            # print(cross_val_score(lgbm_clf, trainX, trainY))
            # [ 0.79016814  0.78542877  0.78350438]
            lgbm_base_trainX, lgbm_base_testX = stackinghelper.get_oof(lgbm_clf, trainX, trainY, testX)
            np.save('basemodels/lgbm_base_trainX', lgbm_base_trainX)
            np.save('basemodels/lgbm_base_testX', lgbm_base_testX)
        else:
            lgbm_base_trainX = np.load('basemodels/lgbm_base_trainX.npy')
            lgbm_base_testX = np.load('basemodels/lgbm_base_testX.npy')

        if not os.path.exists('basemodels/rf_base_trainX.npy') or not os.path.exists(
                'basemodels/rf_base_testX.npy'):
            rf_clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=SEED)
            rf_base_trainX, rf_base_testX = stackinghelper.get_oof(rf_clf, trainX, trainY, testX)
            np.save('basemodels/rf_base_trainX', rf_base_trainX)
            np.save('basemodels/rf_base_testX', rf_base_testX)
        else:
            rf_base_trainX = np.load('basemodels/rf_base_trainX.npy')
            rf_base_testX = np.load('basemodels/rf_base_testX.npy')

        if not os.path.exists('basemodels/knn_base_trainX.npy') or not os.path.exists(
                'basemodels/knn_base_testX.npy'):
            knn_clf = KNeighborsClassifier(n_jobs=-1)
            # print(cross_val_score(knn_clf, trainX, trainY))
            # [0.74553268  0.7424391   0.74456686]
            knn_base_trainX, knn_base_testX = stackinghelper.get_oof(knn_clf, trainX, trainY, testX)
            np.save('basemodels/knn_base_trainX', knn_base_trainX)
            np.save('basemodels/knn_base_testX', knn_base_testX)
        else:
            knn_base_trainX = np.load('basemodels/knn_base_trainX.npy')
            knn_base_testX = np.load('basemodels/knn_base_testX.npy')

        if not os.path.exists('basemodels/xgb_base_trainX.npy') or not os.path.exists(
                'basemodels/xgb_base_testX.npy'):
            xgb_clf = XGBClassifier(n_estimators=1000, seed=SEED)
            # print(cross_val_score(xgb_clf, trainX, trainY))
            # [0.78225138  0.78497624  0.78176879]
            xgb_base_trainX, xgb_base_testX = stackinghelper.get_oof(xgb_clf, trainX, trainY, testX)
            np.save('basemodels/xgb_base_trainX', xgb_base_trainX)
            np.save('basemodels/xgb_base_testX', xgb_base_testX)
        else:
            xgb_base_trainX = np.load('basemodels/xgb_base_trainX.npy')
            xgb_base_testX = np.load('basemodels/xgb_base_testX.npy')

        new_base_trainX = np.hstack(
            (lr_base_trainX, sgd_base_trainX, svm_base_trainX, lgbm_base_trainX, rf_base_trainX, knn_base_trainX,
             xgb_base_trainX))
        new_base_testX = np.hstack(
            (lr_base_testX, sgd_base_testX, svm_base_testX, lgbm_base_testX, rf_base_testX, knn_base_testX,
             xgb_base_testX))
        np.save('basemodels/new_base_trainX', new_base_trainX)
        np.save('basemodels/new_base_testX', new_base_testX)
    else:
        new_base_trainX = np.load('basemodels/new_base_trainX.npy')
        new_base_testX = np.load('basemodels/new_base_testX.npy')

    ori_base_trainX = np.load('basemodels/ori_base_trainX.npy')
    ori_base_testX = np.load('basemodels/ori_base_testX.npy')

    base_trainX = np.hstack((ori_base_trainX, new_base_trainX))
    base_testX = np.hstack((ori_base_testX, new_base_testX))

    stack_lgbm_clf = LGBMClassifier(n_estimators=100, min_child_weight=1, num_leaves=100, colsample_bytree=0.3,
                                    learning_rate=0.0506, reg_alpha=0.78, reg_lambda=0.288,
                                    seed=SEED)

    # {'num_leaves': 25}
    # 0.807889576105

    # {'min_child_weight': 9}
    # 0.806079348318

    # 0.809699803892
    # 0.811962588626

    # 0.812088298889
    # 0.812264293257
    # params = {
    #     'reg_lambda': np.arange(0.285, 0.296, 0.001),
    # }
    # clf = GridSearchCV(stack_lgbm_clf, params, n_jobs=-1, verbose=1000)
    # clf.fit(base_trainX, trainY)
    # print(clf.cv_results_['mean_train_score'])
    # print(clf.cv_results_['mean_test_score'])
    # print(clf.best_params_)
    # print(clf.best_score_)


    # print(cross_val_score(stack_lgbm_clf, base_trainX, trainY, n_jobs=-1, verbose=1000))
    # [0.81376762  0.81175051  0.81127377]


    stack_lgbm_clf.fit(base_trainX, trainY)
    y_pred = stack_lgbm_clf.predict(base_testX)
    y_pred = encoder.inverse_transform(y_pred)

    sub = pd.DataFrame({
        'id': test_df['id'],
        'cuisine': y_pred
    }, columns=['id', 'cuisine'])
    sub.to_csv('output/final_sub_oct7.csv', index=False)
