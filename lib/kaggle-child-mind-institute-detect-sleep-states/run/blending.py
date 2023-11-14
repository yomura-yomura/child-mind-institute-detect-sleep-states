import pathlib

import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import numpy as np
import tqdm

import child_mind_institute_detect_sleep_states.data.comp_dataset

np.seterr(all="raise")

project_root_path = pathlib.Path(__file__).parent.parent

predicted_npz_format = "predicted-fold_{i_fold}.npz"

post_process_modes = {
    # "sleeping_edges_as_probs": cmi_dss_lib.utils.post_process.SleepingEdgesAsProbsSetting(
    #     sleep_prob_th=0.2, min_sleeping_hours=6
    # ),
    "cutting_probs_by_sleep_prob": cmi_dss_lib.utils.post_process.CuttingProbsBySleepProbSetting(
        watch_interval_hour=6, sleep_occupancy_th=0.3
    ),
}


pred_dir_path = project_root_path / "run" / "predicted" / "train"

model_dir_paths = [
    project_root_path
    / "run"
    / "predicted"
    / "jumtras"
    / "exp016-gru-feature-fp16-layer4-ep70-lr-half",
    # project_root_path / "run" / "predicted" / "train" / "exp015-lstm-feature-108-sigma",
    # pred_dir_path / "exp019-stacked-gru-4-layers-24h-duration-4bs-108sigma",
    project_root_path
    / "run"
    / "predicted"
    / "ranchantan"
    / "exp036-stacked-gru-4-layers-24h-duration-4bs-108sigma-with-step-validation",
    pred_dir_path / "exp027-TimesNetFeatureExtractor-1DUnet-Unet",
]


def calc_score(
    i_fold: int, weights: list[int], keys_dict, all_event_df, preds_dict, post_process_modes
):
    series_ids = keys_dict[i_fold]
    # unique_series_ids = np.unique([str(k).split("_")[0] for k in keys])
    unique_series_ids = np.unique(series_ids)
    event_df = all_event_df[all_event_df["series_id"].isin(unique_series_ids)].dropna()

    df_submit_list = []
    for series_id, preds in zip(series_ids, preds_dict[i_fold], strict=True):
        mean_preds = np.average(preds, axis=0, weights=weights)
        df_submit_list.append(
            cmi_dss_lib.utils.post_process.post_process_for_seg(
                #
                preds=mean_preds,
                # preds=corrected_preds[:, :, [1, 2]],
                downsample_rate=2,
                keys=[series_id] * len(mean_preds),
                score_th=0.005,
                distance=96,
                post_process_modes=post_process_modes,
                print_msg=False,
            )
        )
    df_submit = pd.concat(df_submit_list)

    return cmi_dss_lib.utils.metrics.event_detection_ap(
        event_df[event_df["series_id"].isin(unique_series_ids)], df_submit
    )


# scores = calc_all_scores(weights=[1, 1])


if __name__ == "__main__":
    import multiprocessing

    import numpy as np
    import pandas as pd

    # predicted_dict = {
    #     i_fold: {
    #         model_dir_path.name: np.load(
    #             model_dir_path / predicted_npz_format.format(i_fold=i_fold)
    #         )
    #         for model_dir_path in model_dir_paths
    #     }
    #     for i_fold in range(5)
    # }

    predicted_npz_paths = [
        [
            model_dir_path / predicted_npz_format.format(i_fold=i_fold)
            for model_dir_path in model_dir_paths
        ]
        for i_fold in range(5)
    ]  # (fold, model)

    import itertools

    import cmi_dss_lib.utils.common
    import tqdm

    keys_dict = {}
    preds_dict = {}
    for i_fold in tqdm.trange(5):
        cmi_dss_lib.utils.common.save_predicted_npz_group_by_series_id(
            predicted_npz_paths[i_fold], dataset_type="train"
        )
        (
            keys_dict[i_fold],
            preds_dict[i_fold],
        ) = cmi_dss_lib.utils.common.load_predicted_npz_group_by_series_id(
            predicted_npz_paths[i_fold]
        )

        # preds = [data["pred"].reshape(-1, 3) for data in predicted_dict[i_fold].values()]

        # first, *others = predicted_dict[i_fold].values()
        # for other in others:
        #     assert np.all(first["key"] == other["key"])
        #     assert np.all(first["label"] == other["label"])
        # keys_dict[i_fold] = first["key"]
        # preds_dict[i_fold] = np.stack(
        #     [data["pred"] for data in predicted_dict[i_fold].values()], axis=0
        # )

    all_event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")

    # calc_score(0, [1, 1, 1], keys_dict, all_event_df, preds_dict, None)

    def calc_all_scores(weights: list[int], post_process_modes=None):
        scores = []
        for i_fold in tqdm.trange(5):
            scores.append(
                calc_score(
                    i_fold,
                    weights,
                    keys_dict,
                    all_event_df,
                    preds_dict,
                    post_process_modes=post_process_modes,
                )
            )

        mean_score_str, *score_strs = map("{:.3f}".format, [np.mean(scores), *scores])
        print(f"{mean_score_str} ({', '.join(score_strs)})")
        return scores, weights

    def get_grid(step: float, target_sum: float = 1, start: float = 0):
        assert step < 1
        assert 0 <= target_sum

        target_sum *= round(1 / step)
        base_weight = pd.DataFrame(
            np.arange(round(1 / step) + 1) + round(start * round(1 / step)), dtype="i4"
        )

        weight = base_weight.copy()
        for i, _ in enumerate(tqdm.trange(len(model_dir_paths) - 1)):
            weight = weight.merge(base_weight.rename(columns={0: i + 1}), how="cross")
            weight = weight[np.sum(weight, axis=1) <= target_sum].reset_index(drop=True)
        weight = weight.to_numpy()
        weight = weight[np.sum(weight, axis=1) == target_sum]
        print(f"{weight.shape = }")
        return weight * step

    # weight = get_grid(step=0.1)
    weight = get_grid(step=0.1, target_sum=1)
    # weight = get_grid(step=0.02, target_sum=1)

    target_csv_path = (
        pathlib.Path(__file__).parent
        / "grid_search"
        / "_".join(p.name for p in model_dir_paths)
        / "grid_search.csv"
    )

    if target_csv_path.exists():
        df = pd.read_csv(target_csv_path)
        df["scores"] = df["scores"].apply(
            lambda w: [float(n.strip("' ")) for n in w.strip("[]").split(",")]
        )
        df["weights"] = df["weights"].apply(
            lambda w: [float(n.strip("' ")) for n in w.strip("[]").split(",")]
        )

        # if True:
        #     target_weight = df.iloc[df["CV"].argmax()]["weights"]
        #     print(f"{target_weight = }")
        #     new_weight = target_weight + get_grid(0.05, 0, -0.5)
        #     new_weight = new_weight[np.all(new_weight > 0, axis=1)]
        #     assert np.all(np.isclose(np.sum(new_weight, axis=1), 1))
        #     weight = np.concatenate([weight, new_weight], axis=0)

        loaded_weight = np.array(df["weights"].tolist())
        weight = np.array(
            [w for w in weight if not np.any(np.all(np.isclose(w, loaded_weight), axis=1))]
        )
        print(f"-> {weight.shape = }")

        records = df.to_dict("records")
    else:
        records = []

    n_steps_to_save = 30
    with multiprocessing.Pool(8) as p:
        with tqdm.tqdm(total=len(weight), desc="grid search") as t:
            for scores, weights in p.imap_unordered(calc_all_scores, weight.tolist()):
                t.update(1)
                records.append({"CV": np.mean(scores), "scores": scores, "weights": weights})

                if len(records) % n_steps_to_save == 0 or t.n == t.total:
                    df = pd.DataFrame(records)
                    target_csv_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(target_csv_path, index=False)

                    record_at_max = df.iloc[df["CV"].argmax()]
                    print(
                        """
max:
  CV = {CV:.4f}
  weights = {weights}
""".format(
                            **record_at_max.to_dict()
                        )
                    )

            score = calc_all_scores(record_at_max["weights"], post_process_modes)


# scores = [calc_all_scores(weights=w) for w in tqdm.tqdm(weight, desc="grid search")]


# mean_scores = np.mean(scores, axis=1)
# order = np.argsort(mean_scores)[::-1]
# mean_scores[order]
# weight_list[order]

# #
#
# scores = np.array(
#     [
#         [
#             calc_score(i_fold, weights=[w, 1 - w])
#             for w in tqdm.tqdm(weight_list, desc="grid search")
#         ]
#         for i_fold in range(5)
#     ]
# )  # (fold, weight)
#
# weights = weight_list[np.argmax(scores, axis=1)]
# np.mean(np.max(scores, axis=1))
