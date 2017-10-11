import datetime
import os
import pickle as pkl

import keras
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential

import new_neuralnet


def create_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=120))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':

    test_df = pd.read_json('input/test.json')
    ids = test_df['id']

    lgbm_train_x = np.load('origin/basemodels/lgbm_base_train_x.npy')
    lr_train_x = np.load('origin/basemodels/lr_base_train_x.npy')
    nn_train_x = np.load('origin/basemodels/nn_base_train_x.npy')
    rf_train_x = np.load('origin/basemodels/rf_base_train_x.npy')
    sgd_train_x = np.load('origin/basemodels/sgd_base_train_x.npy')
    svm_train_x = np.load('origin/basemodels/svm_base_train_x.npy')

    lgbm_test_x = np.load('origin/basemodels/lgbm_base_test_x.npy')
    lr_test_x = np.load('origin/basemodels/lr_base_test_x.npy')
    nn_test_x = np.load('origin/basemodels/nn_base_test_x.npy')
    rf_test_x = np.load('origin/basemodels/rf_base_test_x.npy')
    sgd_test_x = np.load('origin/basemodels/sgd_base_test_x.npy')
    svm_test_x = np.load('origin/basemodels/svm_base_test_x.npy')

    train_x = np.hstack((lgbm_train_x, lr_train_x, nn_train_x,
                         rf_train_x, sgd_train_x, svm_train_x))
    test_x = np.hstack((lgbm_test_x, lr_test_x, nn_test_x,
                        rf_test_x, sgd_test_x, svm_test_x))

    train_y = np.load('origin/data/train_y.npy')
    encoder = pkl.load(open('origin/data/encoder.pkl', 'rb'))
    train_y = keras.utils.to_categorical(train_y, 20)

    origin_probs = []
    n_estimators = 10
    for i in range(n_estimators):
        model = create_model()
        model_name = 'origin/stacking_nn_weights/model_{}.hdf5'.format(i+1)
        if os.path.exists(model_name):
            model.load_weights(model_name)
        else:
            model.fit(train_x, train_y, batch_size=4096, epochs=100, verbose=1000)
            model.save_weights(model_name)
        origin_probs.append(model.predict_proba(test_x))
    origin_probs = np.array(origin_probs).mean(axis=0)

    new_test_x = np.load('new/data/test_x.npy')
    new_probs = []
    n_estimators = 25
    for i in range(25):
        model_name = 'new/keras/model_{}.hdf5'.format(i+1)
        model = new_neuralnet.create_model()
        model.load_weights(model_name)
        new_probs.append(model.predict_proba(new_test_x))
    new_probs = np.array(new_probs).mean(axis=0)

    probs = origin_probs + new_probs
    y_pred = encoder.inverse_transform(np.argmax(probs, axis=1))

    now = datetime.datetime.now()
    str = now.strftime('%Y-%m-%d_%H-%M-%S')
    sub_name = 'output/sub_{}.csv'.format(str)

    sub = pd.DataFrame({
        'id': ids,
        'cuisine': y_pred
    }, columns=['id', 'cuisine'])
    sub.to_csv(sub_name, index=False)
