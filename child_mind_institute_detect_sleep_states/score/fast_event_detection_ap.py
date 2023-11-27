"""
Forked from https://www.kaggle.com/code/chauyh/kagglechildsleep-fast-ap-metric-computation/notebook
"""
from typing import Literal

import joblib
import numpy as np
import tqdm
from nptyping import Bool, DataFrame, Float, Int, NDArray, Shape
from nptyping import Structure as S

from .event_detection_ap import TOLERANCES


def score(
    event_df: DataFrame[S["series_id: Str, step: Int, event: Str"]],
    sub_df: DataFrame[S["series_id: Str, step: Int, event: Str"]],
    n_jobs: int = -1,
    show_progress=True,
    print_score=True,
) -> float:
    ap_list_dict = {
        event: [metric["average_precision"] for metric in metric_list]
        for event, metric_list in get_score_dict(
            event_df, sub_df, print_score=print_score, n_jobs=n_jobs, show_progress=show_progress
        ).items()
    }
    return np.mean([np.mean(ap_list) for ap_list in ap_list_dict.values()])


def get_score_dict(
    event_df: DataFrame[S["series_id: Str, step: Int, event: Str"]],
    sub_df: DataFrame[S["series_id: Str, step: Int, event: Str"]],
    print_score: bool = True,
    n_jobs: int = -1,
    show_progress: bool = True,
) -> dict[Literal["onset", "wakeup"], list[float]]:
    def calc_metric(series_id: str) -> list[list[tuple]]:
        metrics_dict = {event: [EventMetrics(tol) for tol in TOLERANCES] for event in event_df["event"].unique()}

        for event, metrics_list in metrics_dict.items():
            for metrics in metrics_list:
                target_sub_df = sub_df.query(f"series_id == '{series_id}' & event == '{event}'")
                metrics.add(
                    pred_locs=target_sub_df["step"],
                    pred_probs=target_sub_df["score"],
                    gt_locs=event_df.query(f"series_id == '{series_id}' & event == '{event}'")["step"].to_numpy("i4"),
                )
        return [
            [(metrics.matches, metrics.probs, metrics.num_positive) for metrics in metrics_list]
            for metrics_list in metrics_dict.values()
        ]

    metrics_dict = {event: [EventMetrics(tol) for tol in TOLERANCES] for event in event_df["event"].unique()}

    unique_series_ids_iter = event_df["series_id"].unique()
    if show_progress:
        unique_series_ids_iter = tqdm.tqdm(unique_series_ids_iter, desc="calc score")

    for result_list_list in joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(calc_metric)(series_id) for series_id in unique_series_ids_iter
    ):
        for metrics_list, result_list in zip(metrics_dict.values(), result_list_list, strict=True):
            for metrics, (matches, probs, num_positive) in zip(metrics_list, result_list, strict=True):
                metrics.matches += matches
                metrics.probs += probs
                metrics.num_positive += num_positive

    metric_list_dict = {
        event: [metrics.get_metrics() for metrics in metrics_list] for event, metrics_list in metrics_dict.items()
    }

    if print_score:
        scores = []
        for event, metric_list in metric_list_dict.items():
            score = np.mean([metric["average_precision"] for metric in metric_list])
            print(f"{event} = {score:.3f}")
            scores.append(score)

        print(f"EventDetectionAP = {np.mean(scores):.3f}")

    return metric_list_dict


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
) -> NDArray[Shape['"*"'], Bool]:
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
