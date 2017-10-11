from origin_knn import knn_pred
from origin_lgbm import lgbm_pred
from origin_lr import lr_pred
from origin_nn import nn_pred
from origin_preprocess import load_data
from origin_rf import rf_pred
from origin_sgd import sgd_pred
from origin_svm import svm_pred
from origin_xgb import xgb_pred

if __name__ == '__main__':
    train_x, test_x, train_y, encoder, ids = load_data()
    # lr_pred()
    # sgd_pred()
    # svm_pred()
    # nn_pred(train_x, test_x, train_y)
    # lgbm_pred()
    rf_pred()
    knn_pred()
    xgb_pred()
