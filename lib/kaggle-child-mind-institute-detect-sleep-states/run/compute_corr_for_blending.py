import numpy as np
from blending import all_model_dir_path_dict
from cmi_dss_lib.blending import get_keys_and_preds

exp_names, model_dir_paths = zip(*all_model_dir_path_dict.items())
exp_names = list(map(str, exp_names))
keys_dict, preds_dict = get_keys_and_preds(model_dir_paths)

concat_preds = np.concatenate(
    [np.concatenate(preds_dict[i_fold], axis=1) for i_fold in range(5)], axis=1
)


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

fasfa

from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA

print("clustering")
k_means = KMeans(n_clusters=7)
transformed = k_means.fit_transform(x)
clusters = k_means.predict(x)

print("pca")
pca = IncrementalPCA(n_components=2, batch_size=100)
pca_transformed = pca.fit_transform(transformed)
fig = px.scatter(
    # title=f"fold #{i_fold + 1}",
    x=pca_transformed[:, 0],
    y=pca_transformed[:, 1],
    color=clusters.astype(str),
    text=exp_names,
)
fig.update_layout(width=600, height=600)
fig.show()

# df = pd.DataFrame({"exp": exp_names, "i_cluster": clusters})
# print(df.sort_values(["i_cluster"]))
