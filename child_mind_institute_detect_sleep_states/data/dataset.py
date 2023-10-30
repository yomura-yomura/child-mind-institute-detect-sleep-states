import pathlib

import joblib
import numpy as np

project_root_path = pathlib.Path(__file__).parent.parent.parent
data_dir_path = project_root_path / "data"


SIGMA = 720  # average length of day is 24*60*12 = 17280 for comparison


def gauss(n=SIGMA, sigma=SIGMA * 0.15):
    # guassian distribution function
    r = range(-int(n / 2), int(n / 2) + 1)
    return [1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-float(x) ** 2 / (2 * sigma**2)) for x in r]


def get_prepare_data_df():
    targets, data, ids = joblib.load(data_dir_path / "sleep-critical-point-prepare-data" / "train_data.pkl")

    X = data[0]

    # turn target inds into array
    target_guassian = np.zeros((len(X), 2))
    for s, e in targets[0]:
        st1, st2 = max(0, s - SIGMA // 2), s + SIGMA // 2 + 1
        ed1, ed2 = e - SIGMA // 2, min(len(X), e + SIGMA // 2 + 1)
        target_guassian[st1:st2, 0] = gauss()[st1 - (s - SIGMA // 2) :]
        target_guassian[ed1:ed2, 1] = gauss()[: SIGMA + 1 - ((e + SIGMA // 2 + 1) - ed2)]
