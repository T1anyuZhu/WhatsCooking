import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    # load data
    train_df = pd.read_json('input/train.json', encoding='utf-8')
    test_df = pd.read_json('input/test.json', encoding='utf-8')
    ingredients_train_df = train_df['ingredients']
    ingredients_test_df = test_df['ingredients']
    labels_train_df = train_df['cuisine']

    regex = re.compile('[^A-Za-z0-9]')
    for ingredients in ingredients_train_df:
        for index, ingredient in enumerate(ingredients):
            ingredients[index] = regex.sub('_', ingredient)
    for ingredients in ingredients_test_df:
        for index, ingredient in enumerate(ingredients):
            ingredients[index] = regex.sub('_', ingredient)

    vectorizer = CountVectorizer()
    ingredients_text_list = [' '.join(ingredients) for ingredients in list(ingredients_train_df)]
    train_X = vectorizer.fit_transform(ingredients_text_list).toarray()

    label_encoder = LabelEncoder()
    train_Y = label_encoder.fit_transform(labels_train_df)

    test_X = CountVectorizer(vocabulary=vectorizer.vocabulary_).fit_transform(
        [' '.join(ingredients) for ingredients in list(ingredients_test_df)]).toarray()

    pca = PCA(n_components=2000)
    train_X = pca.fit_transform(train_X)
    test_X = pca.transform(test_X)

    clf = LogisticRegression()
    score = model_selection.cross_val_score(clf, train_X, train_Y, cv=5, n_jobs=-1)
    print(score)

    clf.fit(train_X, train_Y)
    test_Y = clf.predict(test_X)
    test_Y = label_encoder.inverse_transform(test_Y)

    submission = pd.DataFrame({
        "id": test_df['id'],
        "cuisine": test_Y
    })
    submission.to_csv('output/submission.csv', index=False)
