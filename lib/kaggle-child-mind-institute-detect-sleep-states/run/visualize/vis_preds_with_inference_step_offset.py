import numpy as np
import pandas as pd
import tqdm
from vis_preds import Plotter

i_record = 0

inference_step_offsets = np.arange(0, 24, 2) * 12 * 60

i_fold = 0
plotters = [
    Plotter(
        "ranchantan/exp050-transformer-decoder_retry_resume",
        i_fold,
        "valid",
        inference_step_offset=inference_step_offset,
    )
    for inference_step_offset in tqdm.tqdm(inference_step_offsets)
]


df = pd.concat(
    [
        plotter.get_pred_df(plotter.get_indices("1f96b9668bdf", 10)[0])
        .set_index(["step", "type"])["prob"]
        .rename(plotter.cfg.inference_step_offset)
        for plotter in plotters
    ],
    axis=1,
)

import plotly.express as px
import plotly.graph_objs as go

fig = px.line(df.mean(axis=1).rename("mean").reset_index(), x="step", y="mean", color="type")
for event, probs in df.groupby("type"):
    steps = probs.reset_index()["step"].tolist()
    fig.add_trace(
        go.Scatter(
            x=steps + steps[::-1],
            y=probs.min(axis=1).tolist() + probs.max(axis=1).tolist()[::-1],
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
fig.update_layout(width=1500, height=600)
fig.show()


# fig.add_traces(
#     plotters[1]
#     .get_pred_fig(i_record)
#     .for_each_trace(lambda trace: trace.update(x=trace.x + 720))
#     .data
# )
