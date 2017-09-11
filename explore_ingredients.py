import pandas as pd

if __name__ == '__main__':
    train_df = pd.read_json('input/train.json', encoding='utf-8')
    ingredeients_set = set()
    for ingredients in train_df['ingredients']:
        ingredeients_set.update(ingredients)
    print(len(ingredeients_set))
    ingredients_list = list(ingredeients_set)
    ingredients_list.sort(key=lambda x: len(x), reverse=True)
    for ingredient in ingredients_list:
        print(ingredient)
    with open('ingredients.txt', 'w', encoding='utf-8') as f:
        for ingredient in ingredients_list:
            f.write(ingredient + "\n")
