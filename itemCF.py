import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import helpers

def cosine_vector_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = (dot_product / (norm_vec1 * norm_vec2)) if (norm_vec1 * norm_vec2) != 0 else 0
    
    return similarity

def calculate_item_similarity(df):
    items = df.columns[1:]  # Assuming first column is 'UserName'
    item_similarity = pd.DataFrame(index=items, columns=items, dtype=float)
    for item1 in items:
        for item2 in items:
            if item1 == item2:
                item_similarity.loc[item1, item2] = 1.0
            else:
                users_rated_both = helpers.getUsersRatedBoth(df, item1, item2)
                if len(users_rated_both) == 0:
                    item_similarity.loc[item1, item2] = 0.0
                else:
                    ratings1 = df.loc[df['UserName'].isin(users_rated_both), item1].values
                    ratings2 = df.loc[df['UserName'].isin(users_rated_both), item2].values
                    item_similarity.loc[item1, item2] = cosine_vector_similarity(ratings1, ratings2)
    return item_similarity

def predict_ratingItemCF(df, user, item, item_similarity, neighborN=None):
    # fail_value = pd.to_numeric(df.iloc[1:, 1:].stack(), errors='coerce').mean()
    fail_value = .5
    if neighborN is None :
        neighborN = 4
    items_rated_by_user = helpers.getUsersRatedItems(df, user)
    similarities = []
    for rated_item in items_rated_by_user:
        similarity = item_similarity.loc[item, rated_item]
        similarities.append((rated_item, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = similarities[:neighborN]

    numerator = 0
    denominator = 0
    for rated_item, similarity in top_neighbors:
        rating = helpers.getRating(df, user, rated_item)
        numerator += similarity * rating
        denominator += abs(similarity)
    # if denominator == 0:
    #     print('ICF failed')
    # return numerator / denominator if denominator != 0 else 0
    return numerator / denominator if denominator != 0 else fail_value