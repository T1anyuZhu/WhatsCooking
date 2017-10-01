from sklearn import datasets, model_selection
from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import feature_column
import os

from preprocess import load_data
from stackinghelper import StackingHelper, get_oof

if __name__ == '__main__':
    train_df, test_df, trainX, testX, trainY, encoder = load_data()
    feature_columns = [feature_column.real_valued_column("", dimension=1578)]
    classifier = learn.DNNClassifier(hidden_units=[512, 256, 128], feature_columns=feature_columns, n_classes=20,
                                     )
    # train_X, test_X, train_y, test_y = model_selection.train_test_split(trainX, trainY, test_size=0.30, stratify=trainY)
    # classifier.fit(trainX, trainY, steps=2000)
    # acc = classifier.evaluate(test_X, test_y)
    # print(acc)
    # {'loss': 0.77108079, 'accuracy': 0.77021706, 'global_step': 2000}
    # classifier.fit(trainX, trainY)
    if not os.path.exists('stacking/dnn_second_layer'):
        dnn_clf = StackingHelper(classifier)
        dnn_second_layer_trainX, dnn_second_layer_testX = get_oof(dnn_clf, trainX, trainY, testX)
        np.save('stacking/dnn_second_layer_trainX', dnn_second_layer_trainX)
        np.save('stacking/dnn_second_layer_testX', dnn_second_layer_testX)
    else:
        dnn_second_layer_trainX = np.load('stacking/dnn_second_layer_trainX')
        dnn_second_layer_testX = np.load('stacking/dnn_second_layer_testX')
