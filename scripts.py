from collections import defaultdict
import pandas as pd
import numpy as np
from gensim.models import Word2Vec


def get_top_n(predictions, n=100):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def create_id_specific_features(df_train: pd.DataFrame,
                                df_test: pd.DataFrame,
                                group_col: str,
                                agg_cols: dict[str, str],
                                suffix: str,
                                frac: float,
                                seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_grouped = df_train.groupby(group_col).agg(agg_cols)
    common_row = df_grouped.mean().to_dict()
    df_grouped_shuffle_del = df_grouped.sample(
        frac=1-frac, random_state=seed).to_dict("index")
    df_grouped = df_grouped.to_dict("index")
    
    df_train_agg = df_train[group_col].apply(
        lambda id: common_row if df_grouped_shuffle_del.get(id) is None 
                            else df_grouped_shuffle_del.get(id))
    df_test_agg = df_test[group_col].apply(
        lambda id: common_row if df_grouped.get(id) is None 
                            else df_grouped.get(id))
    
    df_train_agg = pd.DataFrame(list(df_train_agg), 
                                index=df_train.index).add_suffix(suffix)
    df_train_agg[group_col] = df_train[group_col]
    df_test_agg = pd.DataFrame(list(df_test_agg),
                               index=df_test.index).add_suffix(suffix)
    return df_train_agg, df_test_agg, common_row


def get_embedding(text:str, vectorizer: Word2Vec) -> np.ndarray:
    """Get Word2Vec embeddings

    Args:
        text (str): text to be vectorized
        vectorizer (Word2Vec): Word2Vec class

    Returns:
        np.ndarray: embedding res
    """
    embeddings = []
    for word in text:
        if word in vectorizer.wv:
            embeddings.append(vectorizer.wv[word])
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(vectorizer.vector_size)
