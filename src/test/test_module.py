import child_mind_institute_detect_sleep_states.data.train
import child_mind_institute_detect_sleep_states.model.multi_res_bi_gru
from child_mind_institute_detect_sleep_states.model.multi_res_bi_gru.config import *
from child_mind_institute_detect_sleep_states.model.multi_res_bi_lstm_attention.model import FOGModel

config = Config(
    dataset=DatasetConfig(
        agg_interval=12,
        features=["max", "min", "mean", "std", "median"],
        sigma=108,
        train_dataset_type="base",
        in_memory=False,
    ),
    train=TrainConfig(fold_type="group", lower_nan_fraction_to_exclude=0.8),
)
df = child_mind_institute_detect_sleep_states.data.train.get_train_df(
    config["dataset"]["sigma"], config["dataset"]["train_dataset_type"]
)
df = df.collect()

data_module = child_mind_institute_detect_sleep_states.model.multi_res_bi_gru.DataModule(df, config, i_fold=0)
data_module.setup("fit")

batch = data_module.train_dataset[0]


model = FOGModel(n_features=2 * len(config["dataset"]["features"]))
model = model.cuda()

inputs = batch[0].cuda()
inputs = inputs.expand(1, *inputs.shape)
outputs = model(inputs)
