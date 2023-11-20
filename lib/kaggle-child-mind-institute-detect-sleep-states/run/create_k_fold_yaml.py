import pathlib

import yaml

import child_mind_institute_detect_sleep_states.data.train
import child_mind_institute_detect_sleep_states.model.multi_res_bi_gru
from child_mind_institute_detect_sleep_states.model.multi_res_bi_gru.config import *

this_dir_path = pathlib.Path(__file__).parent

df = child_mind_institute_detect_sleep_states.data.train.get_train_df(108, None)
df = df.collect()


split_type_dict = {
    "base": dict(lower_nan_fraction_to_exclude=None, lower_repeat_rate_to_exclude=None),
    "0.8-nan": dict(lower_nan_fraction_to_exclude=0.8, lower_repeat_rate_to_exclude=None),
    "0.8-nan_0.6-repeat-rate": dict(lower_nan_fraction_to_exclude=0.8, lower_repeat_rate_to_exclude=0.6),
}

for name, train_config in split_type_dict.items():
    split_dir_path = this_dir_path / "conf" / "split" / name
    split_dir_path.mkdir(exist_ok=True)
    for i_fold in range(5):
        yaml_path = split_dir_path / f"fold_{i_fold}.yaml"
        if yaml_path.exists():
            continue

        data_module = child_mind_institute_detect_sleep_states.model.multi_res_bi_gru.DataModule(
            df,
            Config(
                train=TrainConfig(fold_type="group", **train_config),
                dataset=DatasetConfig(train_dataset_type="base", agg_interval=12, features=[], in_memory=False),
            ),
            i_fold,
        )
        data_module.setup("fit")

        with open(yaml_path, "w") as f:
            yaml.dump(
                {
                    "name": f"fold_{i_fold}",
                    "train_series_ids": data_module.train_dataset.unique_series_ids.tolist(),
                    "valid_series_ids": data_module.valid_dataset.unique_series_ids.tolist(),
                },
                f,
            )
