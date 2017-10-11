import pandas as pd
from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle as pkl

SEED = 2017


def load_data():
    train_df = pd.read_json('input/train.json', encoding='utf-8')
    test_df = pd.read_json('input/test.json', encoding='utf-8')
    ids = test_df['id']

    if not os.listdir('new/data'):

        labels = train_df['cuisine']
        snowball = SnowballStemmer(language='english')
        stpwds = stopwords.words('english')

        train_texts, test_texts = [], []
        for index, (df, texts) in enumerate(zip((train_df, test_df), (train_texts, test_texts))):
            for ingredients in df['ingredients']:
                text = ' '.join(ingredients)
                tokens = word_tokenize(text.lower())
                words = [snowball.stem(token) for token in tokens if token.isalpha() and token not in stpwds]
                texts.append(' '.join(words))

        vectorizer = TfidfVectorizer(ngram_range=(1, 1))
        encoder = LabelEncoder()

        train_x = vectorizer.fit_transform(train_texts).toarray()
        test_x = vectorizer.transform(test_texts).toarray()
        train_y = encoder.fit_transform(labels)

        np.save('new/data/train_x', train_x)
        np.save('new/data/test_x', test_x)
        np.save('new/data/train_y', train_y)
        with open('new/data/encoder.pkl', 'wb') as f:
            pkl.dump(encoder, f)
    else:
        train_x = np.load('new/data/train_x.npy')
        test_x = np.load('new/data/test_x.npy')
        train_y = np.load('new/data/train_y.npy')
        encoder = pkl.load(open('new/data/encoder.pkl', 'rb'))
    return train_x, test_x, train_y, encoder, ids


if __name__ == '__main__':
    load_data()
