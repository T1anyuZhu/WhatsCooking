import re

import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import model_selection
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

sns.set(style="white", context="talk")
from collections import Counter

class StackingEnsembleHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def predict(self, testX):
        return self.clf.predict(testX)

    def fit(self, trainX, trainY):
        return self.clf.fit(trainX, trainY)






def refine(key, repl='_'):
    """
    Extract useful part from ingredient's name
    :param key: ingredient's name
    :type key: str
    :return: refined key
    """

    #  I found some rare characters in training set. First of all, I will use common characters
    #  to substitute them.
    #  e.g. "crème fraîche" --> "creme fraiche"
    rare_character_mapping = {"î": "i", "è": "e", "é": "e", "™": "", "í": "i", "ú": "u", "â": "a", "€": "",
                              "â": "a", "ç": "c", "€": ""}
    for ch in rare_character_mapping:
        key = key.replace(ch, rare_character_mapping[ch])

    # what inside parentheses is supplementary description, not important, just discard.
    #  E.g. "(15 oz.) refried beans" -> "refried beans"
    regex1 = re.compile('\(.*\)')
    rt = regex1.sub('', key.strip())

    #  There are some ingredients like "cream cheese with chives and onion", containing several ingredients.
    #  We need to extract each ingredient.
    #  e.g. "cream cheese with chives and onion" -> ["cream cheese", "chives", "onion"]
    regex2 = re.compile(' and |, | with | ')
    tokens = [token.strip() for token in regex2.split(rt)]

    #  replace numbers and other punctuations with _, actually numbers appeared like "evaporated low-fat 2% milk".
    #  Therefore I think it is not very important. What we are interested are adjectives and nouns.
    regex3 = re.compile('[^A-Za-z- ]')
    tokens = [regex3.sub('', token).lower().strip() for token in tokens]

    wnl = nltk.WordNetLemmatizer()
    lemmatized_words = [wnl.lemmatize(token) for token in tokens if token != '']
    rt = repl.join(lemmatized_words)

    return rt


if __name__ == '__main__':

    #  use pandas to load data from JSON file.
    train_df = pd.read_json('input/train.json', encoding='utf-8')
    test_df = pd.read_json('input/test.json', encoding='utf-8')

    ingredients_counter = Counter()
    for ingredients in train_df['ingredients']:
        ingredients_counter.update(ingredients)

    ingredients_occurence = np.fromiter(iter(ingredients_counter.values()), int)
    ingredients_occurence = np.sort(ingredients_occurence)
    print(np.percentile(ingredients_occurence, list(range(10, 110, 10))))

    for df in (train_df, test_df):
        for index, ingredients in enumerate(df['ingredients']):
            for index, ingredient in enumerate(ingredients):
                ingredients[index] = refine(ingredient, ' ')

    vectorizer = CountVectorizer(stop_words="english")
    encoder = LabelEncoder()

    train_text = [' '.join(ingredients) for ingredients in train_df['ingredients']]
    test_text = [' '.join(ingredients) for ingredients in test_df['ingredients']]

    trainX = vectorizer.fit_transform(train_text).toarray()
    trainY = encoder.fit_transform(train_df['cuisine'])
    testX = np.array(vectorizer.transform(test_text).toarray())

    print(len(vectorizer.vocabulary_))
    for word in vectorizer.vocabulary_:
        print(word)

    ntrain = train_df.shape[0]
    ntest = test_df.shape[0]
    SEED = 0
    NFOLDS = 5
    kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)












    lr_clf = LogisticRegression(max_iter=200)
    score = model_selection.cross_val_score(lr_clf, trainX, trainY, cv=5)
    print(score)


























    lr_clf.fit(trainX, trainY)
    testY = lr_clf.predict(testX)
    testY = encoder.inverse_transform(testY)
    submission = pd.DataFrame({
        'id': test_df['id'],
        'cuisine': testY
    })
    submission.to_csv('output/submission.csv', index=False)






















    #
    #
    # simplify_mapping = {}
    # simplified_ingreidents_counter = {refine(k, simplify_mapping): v for k, v in
    #                                   simplified_ingreidents_counter.items()}
    #
    #
    #
    #
