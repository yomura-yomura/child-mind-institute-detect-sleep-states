import pandas as pd
import plotly.express as px
import polars as pl

import child_mind_institute_detect_sleep_states.model.dataset

feature_names = ["mean", "fft"]
dataset = child_mind_institute_detect_sleep_states.model.dataset.UserWiseDataset(
    pl.scan_parquet("../data/sigma108/fold0/valid.parquet"),
    agg_interval=12,
    feature_names=feature_names,
    in_memory=False,
)
features, uid, steps, labels = dataset[0]
features = features.numpy()

# features[:, 3] = child_mind_institute_detect_sleep_states.model.dataset.features.fft_signal_clean(features[:, 3])

df = pd.DataFrame(
    features[:, features.shape[1] // 2 :],
    columns=[f"{name}({data_type})" for data_type in ["anglez"] for name in feature_names],
)
df["step"] = steps


fig = px.line(df.melt(id_vars="step"), x="step", y="value", facet_row="variable")
fig.show()
