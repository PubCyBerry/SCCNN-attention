from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import pandas as pd


def record_kfold(df, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    train_df = df[df.task == "train"]

    for i, (train, val) in enumerate(skf.split(train_df, train_df.labels), start=1):
        df.loc[train_df.iloc[train].index, f"fold_{i}"] = "train"
        df.loc[train_df.iloc[val].index, f"fold_{i}"] = "valid"
        df[f"fold_{i}"] = df[f"fold_{i}"].fillna("test")

    return df


def get_input_info(inputs: list = ["falff"]) -> dict:
    """
    check input type
    ex)
    inputs = ['wc1', 'falff', 'reho', 'vmhc'] ==> return {'fmri' : ['falff', 'reho', 'vmhc'], 'smri' : ['wc1']}
    inputs = ['falff'] ==> return {'fmri' : ['falff']}
    """
    fMRI_features = [
        "alff",
        "alffg",
        "reho",
        "rehog",
        "vmhc",
        "vmhcg",
        "falff",
        "falffg",
    ]
    sMRI_features = ["c2", "c3", "mwc1", "mwc2", "mwc3", "u", "wc1", "wc2", "wc3"]
    feature_info = dict(fmri=fMRI_features, smri=sMRI_features)

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in feature_info.items()]))
    check = df[df.isin(inputs)]
    features = {
        column: check[column].dropna().tolist()
        for column in check.dropna(axis=1, how="all")
    }

    return features


def extract_cube(data):
    """
    fALFF, ReHo, VMHC = ( 61,  73,  61) -> ( 49,  60,  48) vs (47,  60, 46)
      wc1,  wc2,  wc3 = (121, 145, 121) -> (117, 138, 111) vs (90, 117, 90)
    """
    if data.shape[1] < 100:
        # fmri
        return data[:, 6:55, 7:67, 3:51]
    else:
        # smri
        return data[:, 1:118, 7:145, 0:111]
