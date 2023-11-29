from typing import Literal, cast

import child_mind_institute_detect_sleep_states.data.comp_dataset


def get_duration_dict(phase: Literal["train", "test", "dev"]) -> dict[str, int]:
    dataset_type = cast(Literal["train", "test"], "test" if phase == "test" else "train")
    count_by_series_id_df = (
        child_mind_institute_detect_sleep_states.data.comp_dataset.get_series_df(dataset_type, as_polars=True)
        .group_by("series_id")
        .count()
        .collect()
    )
    return dict(count_by_series_id_df.iter_rows())
