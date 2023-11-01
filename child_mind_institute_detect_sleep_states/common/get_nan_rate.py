import pandas as pd


def get_cat_per_id(
    df: pd.DataFrame, target_cols: str = "series_id", nan_cols: str = "step"
) -> dict[str,float]:
    """get nanrate per user_id """
    dict_result = {}
    for user_id, df in df.groupby(target_cols):
        nans = df[nan_cols].isna().sum()
        lens = df.shape[0]
        nan_rate = nans / lens
        dict_result[str(user_id)] = pd.cut([nan_rate], bins=[i/10 for i in range(11)],labels=[i for i in range(10)])[0]
    return dict_result

