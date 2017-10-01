import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn import DNNClassifier

from preprocess import load_data

if __name__ == '__main__':
    train_df, test_df, trainX, testX, trainY, encoder = load_data()
    feature_columns = [feature_column.real_valued_column("", dimension=1586)]
    dnn_clf = DNNClassifier(hidden_units=[512, 256, 128], feature_columns=feature_columns, n_classes=20,
                            model_dir='tensorflow/test')
    trX, teX, trY, teY = train_test_split(trainX, trainY, test_size=0.2, stratify=trainY)
    dnn_clf.fit(trX, trY, steps=2000)
    print(dnn_clf.evaluate(teX, teY))
