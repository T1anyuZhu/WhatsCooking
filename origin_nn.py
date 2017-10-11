import datetime
import os
import keras
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold

from new_preprocess import load_data, SEED


def create_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=5448))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def nn_pred(train_x, test_x, train_y, kfold=5):
    num_class = np.max(train_y) + 1
    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=SEED)
    base_train_x = np.zeros((train_x.shape[0], num_class))
    base_test_x = np.zeros((kfold, test_x.shape[0], num_class))
    categorical = keras.utils.to_categorical(train_y, 20)
    for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
        print('Fold {}'.format(i + 1))
        model_train_x = train_x[train_index, :]
        model_train_y = categorical[train_index, :]
        model_test_x = train_x[test_index, :]

        train_probs = []
        test_probs = []
        for j in range(10):
            model_name = 'fold{}_model{}.hdf5'.format(i + 1, j + 1)
            model = create_model()
            if os.path.exists('origin/nnweights/{}'.format(model_name)):
                model.load_weights('origin/nnweights/{}'.format(model_name))
            else:
                model.fit(model_train_x, model_train_y, batch_size=4096, epochs=100, verbose=1000)
                model.save_weights('origin/nnweights/{}'.format(model_name))
            train_probs.append(model.predict_proba(model_test_x))
            test_probs.append(model.predict_proba(test_x))
        train_probs = np.array(train_probs).mean(axis=0)
        test_probs = np.array(test_probs).mean(axis=0)
        base_train_x[test_index, :] = train_probs
        base_test_x[i, :, :] = test_probs
    np.save('origin/basemodels/nn_base_train_x', base_train_x)
    np.save('origin/basemodels/nn_base_test_x', base_test_x.mean(axis=0))