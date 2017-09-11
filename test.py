import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

if __name__ == '__main__':
    #  use pandas to load data from JSON file.

    train_df = pd.read_json('input/train.json', encoding='utf-8')
    test_df = pd.read_json('input/test.json', encoding='utf-8')
    train_df.shape, test_df.shape

    #  use pandas to load data from JSON file.

    train_df = pd.read_json('input/train.json', encoding='utf-8')
    test_df = pd.read_json('input/test.json', encoding='utf-8')
    train_df.shape, test_df.shape

    ingredients_counter = Counter()
    for ingredients in train_df['ingredients']:
        ingredients_counter.update(ingredients)
    len(ingredients_counter)

    ingredients_occurence = np.fromiter(iter(ingredients_counter.values()), int)
    ingredients_occurence = np.sort(ingredients_occurence)
    np.percentile(ingredients_occurence, list(range(10, 110, 10)))

    ingredients_occurence = np.fromiter(iter(ingredients_counter.values()), int)
    ingredients_occurence = np.sort(ingredients_occurence)
    np.percentile(ingredients_occurence, list(range(10, 110, 10)))

    simplified_ingreidents_counter = Counter({k: v for k, v in ingredients_counter.items() if v >= 4})
    len(simplified_ingreidents_counter)

    import re

    # tokens = [token.lower() for token in word_tokenize(text)]
    # print(tokens)
    # lemmatized_words = [wnl.lemmatize(token) for token in tokens if not regex.search(wnl.lemmatize(token))]
    # print(lemmatized_words)
    # rt = [word for word in lemmatized_words if word in nouns]
    # print(rt)
    import nltk


    def refine(key):
        """
        Extract useful part from ingredient's name
        :param key: ingredient's name
        :type key: str
        :return: refined key
        """
        #  I found some rare characters in training set. First of all, I will use common characters
        #  to substitute them.
        #  e.g. crème fraîche --> creme fraiche
        rare_character_mapping = {"î": "i", "è": "e", "é": "e", "™": "", "í": "i", "ú": "u", "â": "a", "€": "",
                                  "â": "a", "ç": "c", "€": ""}
        for ch in rare_character_mapping:
            key = key.replace(ch, rare_character_mapping[ch])

        # what inside parentheses is supplementary description, not important, just discard.
        #  E.g. "(15 oz.) refried beans"
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
        tokens = [regex3.sub('', token).strip() for token in tokens]

        wnl = nltk.WordNetLemmatizer()
        lemmatized_words = [wnl.lemmatize(token) for token in tokens]
        rt = '_'.join(lemmatized_words)

        return rt


    simplified_ingreidents_counter = {refine(k): v for k, v in
                                      simplified_ingreidents_counter.items()}
    for k in simplified_ingreidents_counter:
        print(k)


