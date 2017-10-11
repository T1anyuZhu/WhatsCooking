import heapq
import os
import pickle as pkl
from collections import Counter

import numpy as np
import pandas as pd
from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder

SEED = 2017


class Feature(object):
    def __init__(self, ingredient1, ingredient2, supp, conf, cuisine):
        self.ingredient1 = ingredient1
        self.ingredient2 = ingredient2
        self.supp = supp
        self.conf = conf
        self.cuisine = cuisine

    def __lt__(self, other):
        return self.supp * self.conf < other.supp * other.conf

    def __str__(self):
        return '({}, {}) supp:{} conf:{:.4f} type:{}'.format(self.ingredient1, self.ingredient2, self.supp, self.conf,
                                                             self.cuisine)


class FixedMinHeap(object):
    def __init__(self, heap, capacity):
        self.heap = heap
        self.capacity = capacity

    def isempty(self):
        return len(self.heap) == 0

    def put(self, value):
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, value)
        else:
            heapq.heappushpop(self.heap, value)

    def get(self):
        return heapq.heappop(self.heap)


def load_data():
    train_df = pd.read_json('input/train.json', encoding='utf-8')
    test_df = pd.read_json('input/test.json', encoding='utf-8')
    ids = test_df['id']

    if not os.path.exists('origin/data/train_x.npy') or not os.path.exists('origin/data/test_x.npy'):
        snowball = SnowballStemmer(language='english')
        stpwds = stopwords.words('english')

        train_texts, test_texts = [], []
        for index, (df, texts) in enumerate(zip((train_df, test_df), (train_texts, test_texts))):
            for ingredients in df['ingredients']:
                text = ' '.join(ingredients)
                tokens = word_tokenize(text.lower())
                words = [snowball.stem(token) for token in tokens if token.isalpha() and token not in stpwds]
                texts.append(' '.join(words))
        word_counter = Counter()
        for text in train_texts:
            word_counter.update(text.split(' '))

        # occurance = np.array([t[1] for t in word_counter.most_common()])
        # occurance = np.flipud(occurance)
        # print(np.percentile(occurance, np.linspace(0, 100, 10)))
        # [1.00000000e+00   1.00000000e+00   2.00000000e+00   3.00000000e+00
        #  7.00000000e+00   1.30000000e+01   3.10000000e+01   8.80000000e+01
        #  3.32000000e+02   2.71890000e+04]

        vocab = {word for word in word_counter if word_counter[word] >= 5}
        vectorizer = CountVectorizer(vocabulary=vocab)
        encoder = LabelEncoder()
        train_y = encoder.fit_transform(train_df['cuisine'])
        train_x = vectorizer.fit_transform(train_texts).toarray()
        test_x = vectorizer.transform(test_texts).toarray()

        m, n = train_x.shape
        all_recipes = []
        feature_heaps = []

        for i in range(20):
            feature_heaps.append(FixedMinHeap([], 200))

        for index, value in enumerate(encoder.classes_):
            all_recipes.append(train_x[np.where(train_y == index)])

        for i in range(n):
            for j in range(i + 1, n):
                col1 = train_x[:, i]
                col2 = train_x[:, j]
                sum = np.sum(col1 & col2)
                if sum >= 5:
                    conf = np.zeros(20)
                    for index, value in enumerate(all_recipes):
                        recipes = all_recipes[index]
                        col_1 = recipes[:, i]
                        col_2 = recipes[:, j]
                        supp = np.sum(col_1 & col_2)
                        conf[index] = supp
                    supp = np.max(conf)
                    conf = conf / np.sum(conf)
                    if np.max(conf) > 0.5:
                        comb = Feature(i, j, supp, np.max(conf), encoder.inverse_transform(np.argmax(conf)))
                        feature_heaps[np.argmax(conf)].put(comb)
                        print(comb)
        with open('origin/data/combined_features.txt', 'w') as f:
            for index, heap in enumerate(feature_heaps):
                while not heap.isempty():
                    feat = heap.get()
                    f.write(
                        '{},{},{},{},{}\n'.format(feat.ingredient1, feat.ingredient2, feat.supp, feat.conf,
                                                  feat.cuisine))
        combined_trainX, combined_testX = [], []
        with open('origin/data/combined_features.txt', 'r') as f:
            for line in f:
                l = line.strip().split(',')
                col1, col2 = train_x[:, int(l[0])], train_x[:, int(l[1])]
                combined_trainX.append(col1 & col2)
                col1, col2 = test_x[:, int(l[0])], test_x[:, int(l[1])]
                combined_testX.append(col1 & col2)
        combined_trainX = np.array(combined_trainX).transpose()
        combined_testX = np.array(combined_testX).transpose()

        train_x = np.hstack((train_x, combined_trainX))
        test_x = np.hstack((test_x, combined_testX))
        transformer = TfidfTransformer()
        train_x = transformer.fit_transform(train_x).toarray()
        test_x = transformer.transform(test_x).toarray()
        np.save('origin/data/train_x.npy', train_x)
        np.save('origin/data/test_x.npy', test_x)
        np.save('origin/data/train_y.npy', train_y)
        with open('origin/data/encoder.pkl', 'wb') as f:
            pkl.dump(encoder, f)
    else:
        train_x = np.load('origin/data/train_x.npy')
        test_x = np.load('origin/data/test_x.npy')
        train_y = np.load('origin/data/train_y.npy')
        encoder = pkl.load(open('origin/data/encoder.pkl', 'rb'))

    return train_x, test_x, train_y, encoder, ids


if __name__ == '__main__':
    train_x, test_x, train_y, encoder, ids = load_data()
    print('train_x shape: ', train_x.shape)
    print('test_x shape: ', test_x.shape)
