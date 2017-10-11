import datetime
import os
import keras
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from new_preprocess import load_data


def create_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=2570))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    train_x, test_x, train_y, encoder, ids = load_data()

    # n_estimators = 10
    # probs = []
    #
    # for i in range(n_estimators):
    #     model = create_model()
    #     model.fit(train_x, train_y, batch_size=4096, epochs=500)
    #

    train_y = keras.utils.to_categorical(train_y, 20)
    probs = []

    n_estimators = 25
    for i in range(n_estimators):
        model = create_model()
        if os.path.exists('new/keras/model_{}.hdf5'.format(i + 1)):
            model.load_weights('new/keras/model_{}.hdf5'.format(i + 1))
        else:
            model.fit(train_x, train_y, batch_size=4096, epochs=1000, verbose=1000)
            model.save_weights('new/keras/model_{}.hdf5'.format(i + 1), overwrite=True)
        probs.append(model.predict_proba(test_x))

    y_pred = encoder.inverse_transform(np.argmax(np.array(probs).sum(axis=0), axis=1))

    now = datetime.datetime.now()
    str = now.strftime('%Y-%m-%d_%H-%M-%S')
    sub_name = 'new/output/sub_{}.csv'.format(str)

    sub = pd.DataFrame({
        'id': ids,
        'cuisine': y_pred
    }, columns=['id', 'cuisine'])
    sub.to_csv(sub_name, index=False)
