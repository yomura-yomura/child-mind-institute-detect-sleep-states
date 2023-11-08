import pathlib

import numpy as np

project_root_path = pathlib.Path(__file__).parent.parent

predicted_npz_format = "predicted-fold_{i_fold}.npz"

model_dir_paths = [
    # project_root_path / "preds" / "uchida-1",
    project_root_path
    / "cmi-dss-ensemble-models"
    / "jumtras"
    / "exp016-gru-feature-fp16-layer4-ep70-lr-half",
    project_root_path / "output" / "train" / "exp005-lstm-feature-2",
]

predicted_dict = {
    i_fold: {
        model_dir_path.name: np.load(model_dir_path / predicted_npz_format.format(i_fold=i_fold))
        for model_dir_path in model_dir_paths
    }
    for i_fold in range(5)
}

keys_dict = {}
preds_dict = {}
# labels_dict = {}
for i_fold in range(5):
    first, *others = predicted_dict[i_fold].values()
    for other in others:
        assert np.all(first["key"] == other["key"])
        assert np.all(first["label"] == other["label"])
    keys_dict[i_fold] = first["key"]
    # labels_dict[i_fold] = first["label"]
    preds_dict[i_fold] = np.stack(
        [data["pred"] for data in predicted_dict[i_fold].values()], axis=0
    )


post_process_modes = ["adapt_sleep_prob"]

import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import tqdm

import child_mind_institute_detect_sleep_states.data.comp_dataset

all_event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")


def calc_score(i_fold: int, weights: list[int]):
    keys = keys_dict[i_fold]
    unique_series_ids = np.unique([str(k).split("_")[0] for k in keys])
    event_df = all_event_df[all_event_df["series_id"].isin(unique_series_ids)].dropna()

    # labels = labels_dict[i_fold]
    preds = np.average(preds_dict[i_fold], axis=0, weights=weights)
    df_submit = cmi_dss_lib.utils.post_process.post_process_for_seg(
        #
        preds=preds,
        # preds=corrected_preds[:, :, [1, 2]],
        keys=keys,
        score_th=0.005,
        distance=96,
        post_process_modes=post_process_modes,
    ).to_pandas()

    return cmi_dss_lib.utils.metrics.event_detection_ap(
        event_df[event_df["series_id"].isin(unique_series_ids)], df_submit
    )


def calc_all_scores(weights: list[int]):
    scores = []
    for i_fold in tqdm.trange(5):
        scores.append(calc_score(i_fold, weights))

    mean_score_str, *score_strs = map("{:.3f}".format, [np.mean(scores), *scores])
    print(f"{mean_score_str} ({', '.join(score_strs)})")
    return scores


scores = calc_all_scores(weights=[1, 1])

weight_list = np.linspace(0, 1, 20)
scores = [calc_all_scores(weights=[w, 1 - w]) for w in tqdm.tqdm(weight_list, desc="grid search")]


scores = np.array(
    [
        [
            calc_score(i_fold, weights=[w, 1 - w])
            for w in tqdm.tqdm(weight_list, desc="grid search")
        ]
        for i_fold in range(5)
    ]
)  # (fold, weight)

weights = weight_list[np.argmax(scores, axis=1)]
np.mean(np.max(scores, axis=1))
