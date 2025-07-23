import pandas as pd
import numpy as np
import json

def get_personas(data, key, col):
    _data = data.loc[:, [key, col]]
    _data.loc[_data[col].isna(), col] = 'Unknown'
    d = pd.crosstab(_data[key], _data[col])
    return d

def get_relationships(df: pd.DataFrame, group_key: str, json_col: str, count_key: str, title_key: str) -> pd.DataFrame:
    """
    Explodes a column of JSON-encoded lists of dictionaries and extracts count and title fields.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        group_key (str): Column to group by (e.g., 'user_id').
        json_col (str): Column containing JSON strings of list[dict].
        count_key (str): Key to extract count from each dict.
        title_key (str): Key to extract title from each dict.

    Returns:
        pd.DataFrame: Exploded DataFrame with group_key, title, and count columns.
    """
    df_copy = df[[group_key, json_col]].copy()
    df_copy[json_col] = df_copy[json_col].apply(json.loads)
    exploded = df_copy.explode(json_col).dropna(subset=[json_col])
    exploded[title_key] = exploded[json_col].apply(lambda x: x[title_key])
    exploded[count_key] = exploded[json_col].apply(lambda x: x[count_key])
    d = exploded[[group_key, title_key, count_key]]
    return pd.pivot_table(d, index=group_key, columns=title_key, values='count', aggfunc='sum', fill_value=0)

relationships = [
    [{"count": 2, "title": "abc"}, {"count": 1, "title": "bcd"}],
    [{"count": 3, "title": "abc"}],
    [{"count": 1, "title": "abc"}],
    [{"count": 1, "title": "def"}],
]

relationships = [json.dumps(r) for r in relationships]

df = pd.DataFrame({
    'user_id': [1,1,2,3],
    'personas': ['a', 'b', 'a', np.nan],
    'relationships': relationships
})
personas = get_personas(df, 'user_id', 'personas')
relationships = get_relationships(df, 'user_id', 'relationships', 'count', 'title')
