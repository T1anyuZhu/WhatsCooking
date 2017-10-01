import pandas as pd
import numpy as np
from nltk import word_tokenize, RegexpTokenizer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import os


def load_data():
    snowball = SnowballStemmer(language='english')
    stpwds = stopwords.words('english')

    train_df = pd.read_json('input/train.json', encoding='utf-8')
    test_df = pd.read_json('input/test.json', encoding='utf-8')

    if not os.path.exists('data/trainX.npy'):
        if not os.path.exists('pickle/train_text.pkl'):
            train_texts, test_texts = [], []
            for index, (df, texts) in enumerate(zip((train_df, test_df), (train_texts, test_texts))):
                for ingredients in df['ingredients']:
                    text = ' '.join(ingredients)
                    tokens = word_tokenize(text.lower())
                    words = [snowball.stem(token) for token in tokens if token.isalpha() and token not in stpwds]
                    texts.append(' '.join(words))
            with open('pickle/train_text.pkl', 'wb') as f:
                pkl.dump(train_texts, f)
            with open('pickle/test_text.pkl', 'wb') as f:
                pkl.dump(test_texts, f)
            word_counter = Counter()
            for text in train_texts:
                word_counter.update(text.split(' '))
            with open('pickle/word_counter.pkl', 'wb') as f:
                pkl.dump(word_counter, f)
        else:
            with open('pickle/train_text.pkl', 'rb') as f:
                train_text = pkl.load(f)
            with open('pickle/test_text.pkl', 'rb') as f:
                test_text = pkl.load(f)
            with open('pickle/word_counter.pkl', 'rb') as f:
                word_counter = pkl.load(f)

            # occurance = np.array([t[1] for t in word_counter.most_common()])
            # occurance = np.flipud(occurance)
            # print(np.percentile(occurance, np.linspace(0, 100, 10)))
            # [1.00000000e+00   1.00000000e+00   2.00000000e+00   3.00000000e+00
            #  7.00000000e+00   1.30000000e+01   3.10000000e+01   8.80000000e+01
            #  3.32000000e+02   2.71890000e+04]

            vocab = {word for word in word_counter if word_counter[word] >= 5}
            vectorizer = TfidfVectorizer(vocabulary=vocab)
            encoder = LabelEncoder()
            trainY = encoder.fit_transform(train_df['cuisine'])
            trainX = np.array(vectorizer.fit_transform(train_text).toarray())
            testX = np.array(vectorizer.transform(test_text).toarray())
            np.save('data/trainX', trainX)
            np.save('data/testX', testX)
            np.save('data/trainY', trainY)
            with open('pickle/encoder.pkl', 'wb') as f:
                pkl.dump(encoder, f)
    else:
        trainX = np.load('data/trainX.npy')
        testX = np.load('data/testX.npy')
        trainY = np.load('data/trainY.npy')
        with open('pickle/encoder.pkl', 'rb') as f:
            encoder = pkl.load(f)

    return train_df, test_df, trainX, testX, trainY, encoder


if __name__ == '__main__':
    train_df, test_df, trainX, testX, trainY, encoder = load_data()
    print('TrainX shape: {}'.format(trainX.shape))
    print('TestX shape: {}'.format(testX.shape))
