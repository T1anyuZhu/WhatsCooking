import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

class EnsembleHelper(object):

    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)

def simplify_ingredient(ingredient, nouns):
    tokens = [token.lower() for token in word_tokenize(ingredient)]
    lemmatized_ingredient = [wnl.lemmatize(token) for token in tokens if not regex.search(wnl.lemmatize(token))]
    simplified_ingredient = [word for word in lemmatized_ingredient if word in nouns]
    return simplified_ingredient


if __name__ == '__main__':
    # load data
    train_df = pd.read_json('input/train.json', encoding='utf-8')
    test_df = pd.read_json('input/test.json', encoding='utf-8')
    ingredients_train_df = train_df['ingredients']
    ingredients_test_df = test_df['ingredients']
    labels_train_df = train_df['cuisine']

    # count ingredients
    ingredients_set = set()
    ingredients_counter = Counter()

    wnl = WordNetLemmatizer()
    regex = re.compile("\W+")
    nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}

    # for index, ingredients in enumerate(ingredients_train_df):
    #     simplified_ingredients = [simplify_ingredient(ingredient, nouns) for ingredient in ingredients]
    #     ingredients_set.update(simplified_ingredients)
    #     ingredients_counter.update(simplified_ingredients)
    #     for index, ingredient in enumerate(ingredients):
    #         ingredients[index] = simplified_ingredients[index]
    #         simplify_mapper[ingredient] = simplified_ingredients[index]
    # for index, ingredients in enumerate(ingredients_test_df):
    #     for index, ingredient in enumerate(ingredients):
    #         if ingredient in simplify_mapper:
    #             ingredients[index] = simplify_mapper[ingredient]
    #         else:
    #             ingredients[index] = simplify_ingredient(ingredient, nouns)
    # ingredients_set.remove('')
    # ingredients_counter.pop('')

    # for ingredients in ingredients_train_df:
    #     for ingredient in ingredients:
    #         simplified = simplify_ingredient(ingredient, nouns)
    #         ingredients_set.update(simplified)
    #         ingredients_counter.update(simplified)

    # print(len(ingredients_set))  4245

    # ingredients_count_list = np.fromiter(iter(ingredients_counter.values()), dtype=int)
    # ingredients_count_list = np.sort(ingredients_count_list)
    # print(np.percentile(ingredients_count_list, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
    # [1.00000000e+00   1.00000000e+00   2.00000000e+00   3.00000000e+00
    #  5.00000000e+00   9.00000000e+00   1.60000000e+01   3.70000000e+01
    #  1.27000000e+02   1.86620000e+04]

    vectorizer = CountVectorizer()
    label_encoder = LabelEncoder()

    ingredients_text_list = [' '.join(ingredients) for ingredients in list(ingredients_train_df)]
    train_X = np.array(vectorizer.fit_transform(ingredients_text_list).toarray())
    train_Y = np.array(label_encoder.fit_transform(labels_train_df))
    test_X = np.array(vectorizer.transform(
        [' '.join(ingredients) for ingredients in list(ingredients_test_df)]).toarray())

    # m = train_X.shape[0]
    # n = train_X.shape[1]
    # print(m, n)
    # additional_features = np.zeros((m, n * (n - 1) // 2))
    # for i in range(m):
    #     for j in range(n):
    #         if train_X[i, j] != 0:
    #             for k in range(i + 1, n):
    #                 index = n * j - (1 + j) * j // 2 + (k - j) - 1
    #                 additional_features[i, index] = 1 if train_X[i, k] == 1 else 0
    # train_X = np.hstack((train_X, additional_features))
    #
    # m_test = test_X.shape[0]
    # n_test = test_X.shape[1]
    # additional_features_test = np.zeros((m_test, n_test * (n_test - 1) // 2))
    # for i in range(m):
    #     for j in range(n):
    #         if train_X[i, j] != 0:
    #             for k in range(i + 1, n):
    #                 index = n * j - (1 + j) * j // 2 + (k - j) - 1
    #                 additional_features[i, index] = 1 if train_X[i, k] == 1 else 0
    # test_X = np.hstack((test_X, additional_features_test))

    lr_clf = LogisticRegression(max_iter=5000)
    lr_score = model_selection.cross_val_score(lr_clf, train_X, train_Y, cv=5)
    print(lr_score)

    # svm_clf = LinearSVC()
    # svm_score = model_selection.cross_val_score(svm_clf, train_X, train_Y, cv=5)
    # print(svm_score)

    # sgd_clf = SGDClassifier()
    # sgd_score = model_selection.cross_val_score(sgd_clf, train_X, train_Y, cv=5)
    # print(sgd_score)
    #
    # rf_clf = RandomForestClassifier(n_estimators=300)
    # rf_score = model_selection.cross_val_score(rf_clf, train_X, train_Y, cv=5)
    # print(rf_score)
    #
    # knn_clf = KNeighborsClassifier()
    # knn_score = model_selection.cross_val_score(knn_clf, train_X, train_Y, cv=5)
    # print(knn_score)

    lr_clf.fit(train_X, train_Y)

    test_Y = lr_clf.predict(test_X)
    test_Y = label_encoder.inverse_transform(test_Y)
    print(test_Y)

    submission = pd.DataFrame({
        "id": test_df['id'],
        "cuisine": test_Y
    })
    submission.to_csv('output/submission.csv', index=False)
