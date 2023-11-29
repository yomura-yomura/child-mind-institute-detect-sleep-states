from pathlib import Path

path_data = "../../../data/eda"
import numpy as np


def load_predicted(path_data: str):
    dict_preds = {}
    for p in Path(path_data).iterdir():
        for f in p.iterdir():
            data = np.load(f)
            s_id = f.name.split(".")[0]
            dict_preds[s_id] = data["arr_0"]

    return dict_preds


# def load_predicted(path_data:str):
#    series_id = []
#    preds = []
#    for f in range(5):
#        data = np.load(Path(path_data)/ f"predicted-fold_{f}.npz")
#        keys = data["key"]
#        preds.append(data["pred"])
#        series_id.append(np.array(list(map(lambda x: x.split("_")[0], data["key"]))))
#
#
#    series_id = np.concatenate(series_id)
#    preds = np.concatenate(preds)
#    return series_id,preds
