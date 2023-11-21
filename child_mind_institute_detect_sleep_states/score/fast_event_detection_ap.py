"""
Forked from https://www.kaggle.com/code/chauyh/kagglechildsleep-fast-ap-metric-computation/notebook
"""
from typing import Literal

import numpy as np
import tqdm
from nptyping import DataFrame, Float, Int, NDArray, Shape
from nptyping import Structure as S

from .event_detection_ap import TOLERANCES


def get_score_dict(
    event_df: DataFrame[S["series_id: Str, step: Int, event: Str"]],
    sub_df: DataFrame[S["series_id: Str, step: Int, event: Str"]],
    print_score: bool = True,
) -> dict[Literal["onset", "wakeup"], list[float]]:
    metrics_dict = {
        "onset": [EventMetrics(tol) for tol in TOLERANCES],
        "wakeup": [EventMetrics(tol) for tol in TOLERANCES],
    }
    for series_id in tqdm.tqdm(event_df["series_id"].unique(), desc="calc score"):
        for event, metrics_list in metrics_dict.items():
            for metrics in metrics_list:
                target_sub_df = sub_df.query(f"series_id == '{series_id}' & event == '{event}'")
                metrics.add(
                    pred_locs=target_sub_df["step"],
                    pred_probs=target_sub_df["score"],
                    gt_locs=event_df.query(f"series_id == '{series_id}' & event == '{event}'")["step"].to_numpy("i4"),
                )

    ap_list_dict = {
        event: [metrics.get_metrics()["average_precision"] for metrics in metrics_list]
        for event, metrics_list in metrics_dict.items()
    }

    if print_score:
        onset_ap = np.mean(ap_list_dict["onset"])
        wakeup_ap = np.mean(ap_list_dict["wakeup"])
        print(f"{onset_ap = :.3f}")
        print(f"{wakeup_ap = :.3f}")
        print(f"EventDetectionAP = {np.mean([onset_ap, wakeup_ap]):.3f}")

    return ap_list_dict


class EventMetrics:
    def __init__(self, tolerance: int):
        self.tolerance = tolerance

        self.matches = []
        self.probs = []
        self.num_positive = 0

    def add(self, pred_locs: NDArray[Shape['"*"'], Int], pred_probs: NDArray[Shape['"*"'], Float], gt_locs: list[int]):
        matches = match_series(pred_locs, pred_probs, gt_locs, tolerance=self.tolerance)
        self.matches.append(matches)
        self.probs.append(pred_probs)
        self.num_positive += len(gt_locs)

    def get_metrics(self) -> dict:
        matches = np.concatenate(self.matches, axis=0)
        probs = np.concatenate(self.probs, axis=0)

        # sort by probs in descending order
        indices = np.argsort(probs)[::-1]
        matches = matches[indices]
        probs = probs[indices]

        # compute precision and recall curve (using Kaggle code)
        distinct_value_indices = np.where(np.diff(probs))[0]
        threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
        probs = probs[threshold_idxs]

        # Matches become TPs and non-matches FPs as confidence threshold decreases
        tps = np.cumsum(matches)[threshold_idxs]
        fps = np.cumsum(~matches)[threshold_idxs]

        precision = tps / (tps + fps)
        precision[np.isnan(precision)] = 0
        recall = (
            tps / self.num_positive
        )  # total number of ground truths might be different than total number of matches

        # Stop when full recall attained and reverse the outputs so recall is non-increasing.
        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)

        # Final precision is 1 and final recall is 0 and final proba is 1
        precision, recall, probs = np.r_[precision[sl], 1], np.r_[recall[sl], 0], np.r_[probs[sl], 1]

        # compute average precision
        average_precision = -np.sum(np.diff(recall) * np.array(precision)[:-1])

        return {"precision": precision, "recall": recall, "average_precision": average_precision, "prob": probs}

    def reset(self):
        self.matches.clear()
        self.probs.clear()
        self.num_positive = 0


def match_series(
    pred_locs: NDArray[Shape['"*"'], Int],
    pred_probs: NDArray[Shape['"*"'], Float],
    gt_locs: list[int],
    tolerance: int,
) -> NDArray[Shape['"*"'], Int]:
    """
    Probably faster algorithm for matching, since the gt are disjoint (within tolerance)

    pred_locs: predicted locations of events, assume sorted in ascending order
    pred_probs: predicted probabilities of events
    gt_locs: ground truth locations of events (either list[int] or np.ndarray or int32 type)
    """
    assert pred_locs.shape == pred_probs.shape, "pred_locs {} and pred_probs {} must have the same shape".format(
        pred_locs.shape, pred_probs.shape
    )
    assert len(pred_locs.shape) == 1, "pred_locs {} and pred_probs {} must be 1D".format(
        pred_locs.shape, pred_probs.shape
    )
    matches = np.zeros_like(pred_locs, dtype=bool)

    gt_locs = np.asarray(gt_locs, dtype=np.int32)

    # lie within (event_loc - tolerance, event_loc + tolerance), where event_loc in gt_locs
    idx_lows = np.searchsorted(pred_locs, gt_locs - tolerance + 1)
    idx_highs = np.searchsorted(pred_locs, gt_locs + tolerance)
    for k in range(len(gt_locs)):
        idx_low, idx_high = idx_lows[k], idx_highs[k]
        if idx_low == idx_high:
            continue
        # find argmax within range
        max_location = idx_low + np.argmax(pred_probs[idx_low:idx_high])
        matches[max_location] = True
    return matches
