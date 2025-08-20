import pandas as pd
import numpy as np
import math
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
import spacy
import time
import matplotlib.pyplot as plt

def getRating(df, userName, item) :
    return df.loc[df['UserName'] == userName, item].values[0]

def getItemAvg(df, item) :
    return df[item].mean()

def getUsersRatedBoth(df, item1, item2):
    users_item1 = df[df[item1].notnull()]['UserName'].astype(str)
    users_item2 = df[df[item2].notnull()]['UserName'].astype(str)
    both_items = np.intersect1d(users_item1, users_item2)

    return both_items

def getUsersRatedItems(df, user) :
    df_copy = df.copy()
    df_copy.set_index('UserName', inplace=True)
    user_ratings = df_copy.loc[user]
    items_rated_by_user = set(user_ratings[user_ratings.notna()].index)

    return items_rated_by_user

def introduce_sparsity(df, sparsity_level):
    usernames = df['UserName']
    df_without_usernames = df.drop(columns='UserName')
    num_elements = df_without_usernames.size
    current_num_nan = df_without_usernames.isna().sum().sum()
    desired_num_nan = int(sparsity_level * num_elements)
    additional_nan_needed = desired_num_nan - current_num_nan
    sparse_df = df.copy()
    values = sparse_df.drop(columns='UserName').values.flatten()

    if additional_nan_needed > 0:
        non_nan_indices = np.where(~np.isnan(values))[0]
        nan_indices_to_add = np.random.choice(non_nan_indices, additional_nan_needed, replace=False)
        row_counts = np.sum(~np.isnan(sparse_df.drop(columns='UserName').values), axis=1)
        safe_to_nan = np.where(row_counts > 2)[0]

        for index in nan_indices_to_add:
            row_index = index // df_without_usernames.shape[1]
            if row_counts[row_index] > 2:
                values[index] = np.nan
                row_counts[row_index] -= 1
    elif additional_nan_needed < 0:
        nan_indices = np.where(np.isnan(values))[0]
        nan_indices_to_remove = np.random.choice(nan_indices, -additional_nan_needed, replace=False)
        values[nan_indices_to_remove] = df_without_usernames.values.flatten()[nan_indices_to_remove]

    sparse_df.loc[:, sparse_df.columns != 'UserName'] = values.reshape(sparse_df.drop(columns='UserName').shape)
    sparse_df['UserName'] = usernames

    # for index, row in sparse_df.drop(columns='UserName').iterrows():
    #     non_nan_count = row.notna().sum()
    #     if non_nan_count < 2:
    #         original_values = df_without_usernames.loc[index].values
    #         nan_indices_in_row = row.index[row.isna()]
    #         non_nan_to_add = 2 - non_nan_count
    #         # Ensure that nan_indices_in_row are integers
    #         nan_indices_in_row = np.array([df_without_usernames.columns.get_loc(col) for col in nan_indices_in_row])
    #         nan_replacements = np.random.choice(nan_indices_in_row, non_nan_to_add, replace=False)
    #         sparse_df.iloc[index, nan_replacements] = original_values[nan_replacements]

    return sparse_df


# def introduce_item_cold_start_sparsity(df, n_rating_per_item):
    df_data = df.iloc[:, 1:]
    count_non_nan = df_data.notna().sum()
    columns_to_keep = count_non_nan[count_non_nan > n_rating_per_item].index
    modifieddf = df_data[columns_to_keep]
    cssparse_df = pd.DataFrame(index=df.index)
    
    for col in modifieddf.columns:
        non_nan_indices = modifieddf[col].dropna().index
        if len(non_nan_indices) > n_rating_per_item:
            selected_indices = np.random.choice(non_nan_indices, n_rating_per_item, replace=False)
        else:
            selected_indices = non_nan_indices
        
        sparse_col = pd.Series(index=modifieddf.index)
        sparse_col[selected_indices] = modifieddf.loc[selected_indices, col]
        cssparse_df[col] = sparse_col
    
    modifieddf.insert(0, df.columns[0], df.iloc[:, 0])
    cssparse_df.insert(0, df.columns[0], df.iloc[:, 0])
    
    #We need also complete df (named modifieddf), because some columns are removed due to having less rating as the number of required rating. For example, if item cold-start is 2, columns with 1 rating are removed.
    return modifieddf, cssparse_df

def introduce_item_cold_start_sparsity(df, n_rating_per_item):
    df_data = df.iloc[:, 1:]
    
    # Step 1: Keep only items (columns) with enough ratings
    count_non_nan = df_data.notna().sum()
    columns_to_keep = count_non_nan[count_non_nan > n_rating_per_item].index
    filtered_data = df_data[columns_to_keep]

    # Step 2: Drop rows (users) with less than 2 ratings
    filtered_data = filtered_data[filtered_data.notna().sum(axis=1) >= 2]
    
    # Step 3: Initialize cold-start sparse DataFrame
    cssparse_df = pd.DataFrame(index=filtered_data.index)

    # Step 4: For each item (column), keep only n_rating_per_item ratings
    for col in filtered_data.columns:
        non_nan_indices = filtered_data[col].dropna().index
        if len(non_nan_indices) > n_rating_per_item:
            selected_indices = np.random.choice(non_nan_indices, n_rating_per_item, replace=False)
        else:
            selected_indices = non_nan_indices
        
        sparse_col = pd.Series(index=filtered_data.index, dtype=filtered_data[col].dtype)
        sparse_col.loc[selected_indices] = filtered_data.loc[selected_indices, col]
        cssparse_df[col] = sparse_col

    # Step 5: After sparsification, again drop rows with <2 non-NaNs in cssparse_df
    cssparse_df = cssparse_df[cssparse_df.notna().sum(axis=1) >= 2]
    # Ensure modifieddf and cssparse_df have same rows
    filtered_data = filtered_data.loc[cssparse_df.index]

    # Step 6: Reattach ID column
    id_col = df.iloc[:, 0]
    modifieddf = filtered_data.copy()
    cssparse_df = cssparse_df.copy()
    modifieddf.insert(0, df.columns[0], id_col.loc[modifieddf.index])
    cssparse_df.insert(0, df.columns[0], id_col.loc[cssparse_df.index])

    #We need also complete df (named modifieddf), because some columns are removed due to having less rating as the number of required rating. For example, if item cold-start is 2, columns with 1 rating are removed.
    return modifieddf, cssparse_df


'''def introduce_user_cold_start_sparsity(df, n_rating_per_user):
    df_data = df.iloc[:, 1:]  # Exclude the 'UserName' column
    count_non_nan = df_data.notna().sum(axis=1)  # Count non-NaN values per user
    
    rows_to_keep = count_non_nan[count_non_nan > n_rating_per_user].index
    modifieddf = df_data.loc[rows_to_keep]  # Users with more than n_rating_per_user ratings
    cssparse_df = pd.DataFrame(index=df.index, columns=df.columns[1:])  # Initialize cssparse_df with the same columns as df_data
    
    for row_idx in modifieddf.index:
        non_nan_indices = modifieddf.loc[row_idx].dropna().index
        if len(non_nan_indices) > n_rating_per_user:
            selected_indices = np.random.choice(non_nan_indices, n_rating_per_user, replace=False)
        else:
            selected_indices = non_nan_indices
        
        sparse_row = pd.Series(index=modifieddf.columns)  # Ensure sparse_row matches the number of columns in modifieddf
        sparse_row[selected_indices] = modifieddf.loc[row_idx, selected_indices]  # Fill only the selected indices with ratings
        
        # Assign the sparse_row to the corresponding row in cssparse_df
        cssparse_df.loc[row_idx] = sparse_row.values
    
    modifieddf.insert(0, df.columns[0], df.iloc[:, 0])
    cssparse_df.insert(0, df.columns[0], df.iloc[:, 0])
    
    return modifieddf, cssparse_df'''

def introduce_user_cold_start_sparsity(df, n_rating_per_user):
    df_data = df.iloc[:, 1:]  # Exclude the 'UserName' column
    count_non_nan = df_data.notna().sum(axis=1)  # Count non-NaN values per user
    
    rows_to_keep = count_non_nan[count_non_nan > n_rating_per_user].index
    modifieddf = df_data.loc[rows_to_keep].copy()  # Users with more than n_rating_per_user ratings
    cssparse_df = pd.DataFrame(index=rows_to_keep, columns=df.columns[1:])  # Match columns of df_data

    for row_idx in modifieddf.index:
        non_nan_indices = modifieddf.loc[row_idx].dropna().index
        if len(non_nan_indices) > n_rating_per_user:
            selected_indices = np.random.choice(non_nan_indices, n_rating_per_user, replace=False)
        else:
            selected_indices = non_nan_indices
        
        sparse_row = pd.Series(index=modifieddf.columns, dtype=float)
        sparse_row[selected_indices] = modifieddf.loc[row_idx, selected_indices]
        cssparse_df.loc[row_idx] = sparse_row.values

    # Drop rows from both dfs where sparse version has < n_rating_per_user ratings
    valid_rows = cssparse_df.notna().sum(axis=1) >= n_rating_per_user
    cssparse_df = cssparse_df.loc[valid_rows]
    modifieddf = modifieddf.loc[valid_rows]

    # Insert UserName column back
    modifieddf.insert(0, df.columns[0], df.loc[modifieddf.index, df.columns[0]])
    cssparse_df.insert(0, df.columns[0], df.loc[cssparse_df.index, df.columns[0]])

    return modifieddf, cssparse_df

def getHCBCFiSimCF(df, item1, item2) :
    usersRatedBoth = getUsersRatedBoth(df, item1, item2)
    if (len(usersRatedBoth) == 0) : return 0

    sum = 0
    for u in usersRatedBoth :
        predicted = getItemAvg(df, item1) + getRating(df, u, item2) - getItemAvg(df, item2)
        sum += (predicted - getRating(df, u, item1)) ** 2
    nUsersRatedItem1 = len(df[df[item1].notnull()]['UserName'])
    nUsersRatedItem2 = len(df[df[item2].notnull()]['UserName'])

    return (1 - (sum/len(usersRatedBoth))) * len(usersRatedBoth) / math.sqrt(nUsersRatedItem1 * nUsersRatedItem2)

# Function to calculate prediction coverage
def calculate_prediction_coverage(df, sparse_df):
    total_items = df.columns[1:]  # Exclude 'UserName' column
    predicted_items = sparse_df.notna().sum(axis=0)  # Count non-NaN values (predictions made)
    
    prediction_coverage = len(predicted_items[predicted_items > 0]) / len(total_items)
    return prediction_coverage

# Function to calculate weights (utility) based on item popularity (or other criteria)
def calculate_item_weights(df):
    # Utility here is the number of ratings an item has (you can change this to another metric)
    weights = df.notna().sum(axis=0)  # Count the number of non-NaN values for each item (popularity)
    return weights

# Function to calculate weighted prediction coverage
def calculate_weighted_prediction_coverage(df, sparse_df):
    # Get the total set of items and the items for which we have predictions
    total_items = df.columns[1:]  # Exclude 'UserName' column
    item_weights = calculate_item_weights(df)  # Calculate weights for all items
    
    # Calculate weighted coverage
    predicted_items_weights = item_weights[sparse_df.notna().sum(axis=0) > 0]  # Items where predictions are made
    total_weights_sum = item_weights.sum()  # Sum of weights of all items
    
    weighted_coverage = predicted_items_weights.sum() / total_weights_sum
    return weighted_coverage

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in words]

    return ' '.join(words)
