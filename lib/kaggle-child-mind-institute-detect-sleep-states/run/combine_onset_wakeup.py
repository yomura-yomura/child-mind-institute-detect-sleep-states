import pathlib

import numpy as np

project_root_path = pathlib.Path(__file__).parent.parent

predicted_dir_path = project_root_path / "run" / "predicted"


path_dict = {
    "onset": predicted_dir_path
    / "ranchantan"
    / "exp050-transformer-decoder_retry_resume"
    / "train",
    "wakeup": predicted_dir_path / "ranchantan" / "exp075-wakeup_6" / "train",
}

target_pred_dir_path = predicted_dir_path / "combined" / "exp050_exp75-wakeup" / "train"


for i_fold in range(5):
    onset_stems, wakeup_stems = [
        set(p.stem for p in (path_dict[k] / f"fold_{i_fold}").glob("*.npz"))
        for k in ["onset", "wakeup"]
    ]
    assert onset_stems == wakeup_stems
    series_ids = list(onset_stems)

    target_pred_fold_dir_path = target_pred_dir_path / f"fold_{i_fold}"
    target_pred_fold_dir_path.mkdir(exist_ok=True, parents=True)

    for series_id in series_ids:
        base_preds = np.load(path_dict["onset"] / f"fold_{i_fold}" / f"{series_id}.npz")["arr_0"]
        assert base_preds.shape[-1] == 3
        wakeup_preds = np.load(path_dict["wakeup"] / f"fold_{i_fold}" / f"{series_id}.npz")[
            "arr_0"
        ]
        assert wakeup_preds.shape[-1] == 2
        base_preds[:, 2] = wakeup_preds[:, 1]
        np.savez_compressed(target_pred_fold_dir_path / f"{series_id}.npz", base_preds)
