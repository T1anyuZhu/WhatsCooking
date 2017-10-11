import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from origin_preprocess import load_data

SEED = 2017  # for reproducibility

if __name__ == '__main__':
    train_df, test_df, trainX, testX, trainY, encoder = load_data()

    # if not os.path.exists('stacking/lr_second_layer_trainX.npy'):
    lr_clf = LogisticRegression(n_jobs=-1, random_state=SEED)
    params = {
        'C': np.arange(1, 11, 1),
    }
    clf = GridSearchCV(lr_clf, params, n_jobs=-1, verbose=1000)
    clf.fit(trainX, trainY)
    print(clf.cv_results_['mean_train_score'])
    print(clf.cv_results_['mean_test_score'])
    print(clf.best_params_)
    print(clf.best_score_)
    # lr_second_layer_trainX, lr_second_layer_testX = get_oof(lr_clf, trainX, trainY, testX)
    # np.save('stacking/lr_second_layer_trainX', lr_second_layer_trainX)
    # np.save('stacking/lr_second_layer_testX', lr_second_layer_testX)
    # [ 0.78571967  0.78452372  0.78531542]
    # else:
    #     lr_second_layer_trainX = np.load('stacking/lr_second_layer_trainX.npy')
    #     lr_second_layer_testX = np.load('stacking/lr_second_layer_testX.npy')

    # if not os.path.exists('stacking/lgbm_second_layer_trainX.npy'):
    #     lgbm_clf = StackingHelper(LGBMClassifier(n_estimators=1500, colsample_bytree=0.5, learning_rate=0.1, seed=SEED))
    #     lgbm_second_layer_trainX, lgbm_second_layer_testX = get_oof(lgbm_clf, trainX, trainY, testX)
    #     np.save('stacking/lgbm_second_layer_trainX', lgbm_second_layer_trainX)
    #     np.save('stacking/lgbm_second_layer_testX', lgbm_second_layer_testX)
    #     # [ 0.77712433  0.7778113   0.77550558]
    # else:
    #     lgbm_second_layer_trainX = np.load('stacking/lgbm_second_layer_trainX.npy')
    #     lgbm_second_layer_testX = np.load('stacking/lgbm_second_layer_testX.npy')
    #
    # if not os.path.exists('stacking/sgd_second_layer_trainX.npy'):
    #     sgd_clf = StackingHelper(SGDClassifier(n_iter=17, n_jobs=-1, random_state=SEED))
    #     sgd_second_layer_trainX, sgd_second_layer_testX = get_oof(sgd_clf, trainX, trainY, testX)
    #     np.save('stacking/sgd_second_layer_trainX', sgd_second_layer_trainX)
    #     np.save('stacking/sgd_second_layer_testX', sgd_second_layer_testX)
    #     # [0.77079092  0.77042009  0.77195895]
    # else:
    #     sgd_second_layer_trainX = np.load('stacking/sgd_second_layer_trainX.npy')
    #     sgd_second_layer_testX = np.load('stacking/sgd_second_layer_testX.npy')
    #
    # if not os.path.exists('stacking/knn_second_layer_trainX.npy'):
    #     knn_clf = StackingHelper(KNeighborsClassifier(n_jobs=-1))
    #     knn_second_layer_trainX, knn_second_layer_testX = get_oof(knn_clf, trainX, trainY, testX)
    #     np.save('stacking/knn_second_layer_trainX', knn_second_layer_trainX)
    #     np.save('stacking/knn_second_layer_testX', knn_second_layer_testX)
    #     # [0.73143331  0.72667622  0.73302143]
    # else:
    #     knn_second_layer_trainX = np.load('stacking/knn_second_layer_trainX.npy')
    #     knn_second_layer_testX = np.load('stacking/knn_second_layer_testX.npy')
    #
    # if not os.path.exists('stacking/rf_second_layer_trainX.npy'):
    #     rf_clf = StackingHelper(RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=SEED))
    #     rf_second_layer_trainX, rf_second_layer_testX = get_oof(rf_clf, trainX, trainY, testX)
    #     np.save('stacking/rf_second_layer_trainX', rf_second_layer_trainX)
    #     np.save('stacking/rf_second_layer_testX', rf_second_layer_testX)
    #     # [0.75103672  0.74522966  0.74660429]
    # else:
    #     rf_second_layer_trainX = np.load('stacking/rf_second_layer_trainX.npy')
    #     rf_second_layer_testX = np.load('stacking/rf_second_layer_testX.npy')
    #
    # if not os.path.exists('stacking/svm_second_layer_trainX.npy'):
    #     svm_clf = StackingHelper(LinearSVC(random_state=SEED))
    #     svm_second_layer_trainX, svm_second_layer_testX = get_oof(svm_clf, trainX, trainY, testX)
    #     np.save('stacking/svm_second_layer_trainX', svm_second_layer_trainX)
    #     np.save('stacking/svm_second_layer_testX', svm_second_layer_testX)
    #     # [0.78406092  0.7815069   0.78569273]
    # else:
    #     svm_second_layer_trainX = np.load('stacking/svm_second_layer_trainX.npy')
    #     svm_second_layer_testX = np.load('stacking/svm_second_layer_testX.npy')
    #
    # if not os.path.exists('stacking/xgb_second_layer_trainX.npy'):
    #     xgb_clf = StackingHelper(XGBClassifier(n_estimators=2000, seed=SEED))
    #     xgb_second_layer_trainX, xgb_second_layer_testX = get_oof(xgb_clf, trainX, trainY, testX)
    #     np.save('stacking/xgb_second_layer_trainX.npy', xgb_second_layer_trainX)
    #     np.save('stacking/xgb_second_layer_testX', xgb_second_layer_testX)
    # else:
    #     xgb_second_layer_trainX = np.load('stacking/xgb_second_layer_trainX.npy')
    #     xgb_second_layer_testX = np.load('stacking/xgb_second_layer_testX.npy')
    #
    # second_layer_trainX = np.hstack(
    #     (lr_second_layer_trainX, lgbm_second_layer_trainX, sgd_second_layer_trainX, knn_second_layer_trainX,
    #      rf_second_layer_trainX, svm_second_layer_trainX, xgb_second_layer_trainX))
    # second_layer_trainY = trainY
    # second_layer_testX = np.hstack(
    #     (lr_second_layer_testX, lgbm_second_layer_testX, sgd_second_layer_testX, knn_second_layer_testX,
    #      rf_second_layer_testX, svm_second_layer_testX, xgb_second_layer_testX))
    # np.save('basemodels/ori_base_trainX', second_layer_trainX)
    # np.save('basemodels/ori_base_testX', second_layer_testX)

    # second_layer_lgbm = LGBMClassifier(num_leaves=84, learning_rate=0.019849950000000002, n_estimators=100,
    #                                    min_child_samples=13, min_child_weight=1, colsample_bytree=0.55,
    #                                    reg_alpha=0.14594020408163264, reg_lambda=0.096959183673469396, seed=SEED)
    # second_layer_lgbm.fit(second_layer_trainX, second_layer_trainY)
    # print(cross_val_score(second_layer_lgbm, second_layer_trainX, second_layer_trainY))

    # params = {
    # 'num_leaves': np.arange(80, 101, 2),
    # 'min_child_samples': np.arange(10, 21, 1),
    # 'learning_rate': np.linspace(0.01984995, 0.01984996, num=20),
    # 'min_child_weight': np.arange(1, 11, 2),
    # 'reg_alpha': np.linspace(0, 0.01),
    # 'subsample_freq': np.arange(1, 6),
    # 'reg_alpha': np.linspace(0.14594, 0.14595),
    # 'reg_lambda': np.linspace(0.095, 0.097),
    # 'colsample_bytree': np.arange(0.5, 0.61, 0.01),
    # 'min_child_samples': np.linspace(5, 15, dtype=int),
    # }
    #
    # clf = GridSearchCV(second_layer_lgbm, params)
    # clf.fit(second_layer_trainX, second_layer_trainY)
    # score = cross_val_score(second_layer_lgbm, second_layer_trainX, second_layer_trainY)
    # print(score)
    # print(clf.best_score_)
    # print(clf.best_params_)
    #
    # y_pred = second_layer_lgbm.predict(second_layer_testX)
    # y_pred = encoder.inverse_transform(y_pred)
    #
    # sub = pd.DataFrame({
    #     'id': test_df['id'],
    #     'cuisine': y_pred
    # }, columns=['id', 'cuisine'])
    # sub.to_csv('output/stacking_submission_final.csv', index=False)
