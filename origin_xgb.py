import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from origin_preprocess import load_data, SEED
from stacking_utils import get_base_predictions

def xgb_pred():
    train_x, test_x, train_y, encoder, ids = load_data()
    xgb_clf = XGBClassifier(n_estimators=1000, seed=SEED)
    print(xgb_clf.__class__.__name__)
    # print(cross_val_score(xgb_clf, train_x, train_y, n_jobs=-1, verbose=1000))

    base_train_x, base_test_x = get_base_predictions(xgb_clf, train_x, test_x, train_y, kfold=5)
    np.save('origin/basemodels/xgb_base_train_x', base_train_x)
    np.save('origin/basemodels/xgb_base_test_x', base_test_x)
