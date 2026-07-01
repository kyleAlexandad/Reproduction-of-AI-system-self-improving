import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    n_comps = int(min(20, X.shape[1] - 1, X.shape[0] - 1))
    emb = PCA(n_components=n_comps, random_state=0).fit_transform(X)
    adata.obsm["X_emb"] = emb
    return adata
