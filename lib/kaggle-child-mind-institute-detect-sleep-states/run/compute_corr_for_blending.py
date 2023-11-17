import numpy as np
from blending import all_model_dir_path_dict, get_keys_and_preds

exp_names, model_dir_paths = zip(*all_model_dir_path_dict.items())
exp_names = list(map(str, exp_names))
keys_dict, preds_dict = get_keys_and_preds(model_dir_paths)

i_fold = 0
stacked_preds = np.concatenate(preds_dict[i_fold], axis=1)
assert stacked_preds.ndim == 3  # (model, duration, pred_type)
x = stacked_preds[..., [1, 2]].reshape(stacked_preds.shape[0], -1)
corr_coefficient = np.corrcoef(x)

import plotly.express as px

px.imshow(
    corr_coefficient, x=exp_names, y=exp_names, text_auto=".2f", title=f"fold {i_fold + 1}"
).show()

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA

print("clustering")
k_means = KMeans(n_clusters=7)
transformed = k_means.fit_transform(x)
clusters = k_means.predict(x)

print("pca")
pca = IncrementalPCA(n_components=2, batch_size=100)
pca_transformed = pca.fit_transform(transformed)
px.scatter(
    x=pca_transformed[:, 0], y=pca_transformed[:, 1], color=clusters.astype(str), text=exp_names
).show()

df = pd.DataFrame({"exp": exp_names, "i_cluster": clusters})
print(df.sort_values(["i_cluster"]))
