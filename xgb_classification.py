from sklearn import metrics

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np

from preprocess import load_data
from stackinghelper import StackingHelper


def modelfit(clf, trainX, trainY, useTrainCV=True, cv_fold=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_params = clf.get_xgb_params()
        xgtrain = xgb.DMatrix(trainX, trainY)
        cvresult = xgb.cv(xgb_params, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_fold,
                          metrics='accuracy', early_stopping_rounds=early_stopping_rounds)
        clf.set_params(n_estimators=cvresult.shpae[0])
    clf.fit(trainX, trainY)
    y_pred = clf.predict(trainX)
    y_pred_prob = clf.predict_proba(trainX)[:, 1]
    print('Model Report')
    print('Accuracy: {:.4f}'.format(metrics.accuracy_score(trainY, y_pred)))


if __name__ == '__main__':
    train_df, test_df, trainX, testX, trainY, encoder = load_data()

    xgb_clf = XGBClassifier(n_estimators=2000)
    # params = {
    #     'max_depth': np.arange(3, 10, 2),
    #     'min_child_weight': np.arange(1, 6, 2)
    # }
    # clf = GridSearchCV(xgb_clf, params, verbose=1)
    # clf.fit(trainX, trainY)
    # print(clf.cv_results_)
    clf = StackingHelper(xgb_clf)

    # mlp_clf = MLPClassifier()
    # print(cross_val_score(mlp_clf, trainX, trainY))
