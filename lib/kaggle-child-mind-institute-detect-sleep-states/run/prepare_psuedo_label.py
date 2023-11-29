from pathlib import Path

import pandas as pd
from cmi_dss_lib.utils.load_predicted import load_predicted
from cmi_dss_lib.utils.post_process import post_process_for_seg
from tqdm import tqdm


def preprocess_for_pseudo_labels(dict_data, result_path: str, score_th: float = 0.5) -> None:
    """submitファイルのscore_th以上のラベルを抽出
    pseudo_labeling v1で使用
    """

    list_df = []
    for series_id, preds in tqdm(dict_data.items(), desc="convert to label"):
        df = post_process_for_seg(
            keys=[series_id] * len(preds),
            preds=preds,
            labels=["sleep", "event_onset", "event_wakeup"],
            # preds=corrected_preds[:, :, [1, 2]],
            downsample_rate=2,
            score_th=0.005,
            distance=96,
            post_process_modes=None,
            print_msg=False,
        ).drop(columns=["row_id"])
        list_df.append(df.query("score >= @score_th"))

    pd.concat(list_df).to_csv(f"{result_path}/pseudo_label_{score_th}.csv", index=None)
    return


if __name__ == "__main__":
    # 予測値のパスを指定
    path_data = ""

    dict_data = load_predicted(path_data)

    # 保存先を指定
    result_path = Path(__file__).parent.parent / "pseudo_label"

    result_path.mkdir(exist_ok=True, parents=True)
    preprocess_for_pseudo_labels(dict_data=dict_data, result_path=result_path)
