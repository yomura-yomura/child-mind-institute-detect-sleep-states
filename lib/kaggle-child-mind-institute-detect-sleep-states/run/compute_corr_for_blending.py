import numpy as np
import pandas as pd
import tqdm
from blending import all_model_dir_path_dict
from cmi_dss_lib.blending import get_keys_and_preds

exp_names, model_dir_paths = zip(*all_model_dir_path_dict.items())
exp_names = list(map(str, exp_names))
keys_dict, preds_dict = get_keys_and_preds(model_dir_paths)


# concat_preds = np.concatenate(
#     [np.concatenate(preds_dict[i_fold], axis=1) for i_fold in range(5)], axis=1
# )

n = sum(preds.shape[1] for preds_list in preds_dict.values() for preds in preds_list)
concat_preds = np.empty((len(all_model_dir_path_dict), n, 3), dtype="f2")
print(f"{concat_preds.nbytes / 1024 ** 3 = :.2f} GB")
start = 0
for preds_list in tqdm.tqdm(preds_dict.values()):
    for preds in preds_list:
        end = start + preds.shape[1]
        concat_preds[:, start:end, :] = preds
        start = end


assert concat_preds.ndim == 3  # (model, duration, pred_type)
x = concat_preds[..., [1, 2]].reshape(concat_preds.shape[0], -1)
corr_coefficient = np.corrcoef(x)

import plotly.express as px

fig = px.imshow(
    corr_coefficient,
    # title=f"fold #{i_fold + 1}",
    x=exp_names,
    y=exp_names,
    text_auto=".2f",
)
fig.show()


from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA

print("clustering")
k_means = MiniBatchKMeans(n_clusters=8, batch_size=10)
k_means.fit(x)
transformed = k_means.transform(x)
clusters = k_means.predict(x)

print("pca")
pca = IncrementalPCA(n_components=2, batch_size=10)
pca_transformed = pca.fit_transform(transformed)
fig = px.scatter(
    # title=f"fold #{i_fold + 1}",
    x=pca_transformed[:, 0],
    y=pca_transformed[:, 1],
    color=clusters.astype(str),
    text=exp_names,
)
fig.update_layout(width=600, height=600)
fig.update_traces(textposition="bottom center")
fig.show()

score_dict = {
    3: 0.7513,
    7: 0.7464,
    19: 0.7610,
    27: 0.7610,
    41: 0.7673,
    47: 0.7718,
    50: 0.7701,
    52: 0.7690,
    53: 0.7748,
    54: 0.7544,
    55: 0.7471,
    58: 0.7732,
    60: 0.7688,
    73: 0.7712,
}

scores = [score_dict[k] for k in all_model_dir_path_dict.keys()]

df = pd.DataFrame({"exp": exp_names, "score": scores, "cluster_id": clusters})
df["exp"] = df["exp"].astype(int)
df = df.sort_values(["cluster_id", "score"], ascending=[True, False])
print(df)
print(df.groupby("cluster_id").head(1).sort_values("exp"))
