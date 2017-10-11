from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from origin_preprocess import load_data
from stacking_utils import get_base_predictions


def knn_pred():
    train_x, test_x, train_y, encoder, ids = load_data()
    knn_clf = KNeighborsClassifier(weights='distance')
    print(knn_clf.__class__.__name__)
    # print(cross_val_score(knn_clf, train_x, train_y, n_jobs=-1))
    base_train_x, base_test_x = get_base_predictions(knn_clf, train_x, test_x, train_y, kfold=5)
    np.save('origin/basemodels/knn_base_train_x', base_train_x)
    np.save('origin/basemodels/knn_base_test_x', base_test_x)
