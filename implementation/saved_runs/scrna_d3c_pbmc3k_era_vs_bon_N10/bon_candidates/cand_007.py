import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata_copy = adata.copy()

    # 1. Standard preprocessing
    sc.pp.normalize_total(adata_copy, target_sum=1e4)
    sc.pp.log1p(adata_copy)

    # 2. Optional: Scale data to prevent highly expressed genes from dominating
    #    sc.pp.scale automatically converts to dense if zero_center=True (default)
    #    and max_value clips extreme values.
    sc.pp.scale(adata_copy, max_value=10)

    # Ensure data is dense for PCA
    X = adata_copy.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32) # Ensure float type for PCA

    # 3. Perform PCA
    # Determine number of components robustly
    n_comps = int(min(20, X.shape[1] - 1, X.shape[0] - 1))
    if n_comps < 1: # Handle edge case where too few features/samples for PCA
        # If PCA is not possible, return an identity or very low-dim embedding
        # as a fallback, ensuring the output format is correct.
        # For simplicity and to satisfy 1 <= d <= n_genes, we can choose 1.
        adata_copy.obsm["X_emb"] = np.zeros((adata_copy.n_obs, 1), dtype=np.float32)
        return adata_copy

    pca = PCA(n_components=n_comps, random_state=0)
    emb_pca = pca.fit_transform(X)

    # 4. Subtract each BATCH's MEAN vector in EMBEDDING space (batch correction)
    corrected_emb = np.copy(emb_pca)
    batches = adata_copy.obs["batch"].to_numpy() # Get batch labels as numpy array

    for batch_label in np.unique(batches):
        batch_mask = (batches == batch_label)
        if np.sum(batch_mask) > 0: # Ensure there are cells in the batch
            batch_mean = emb_pca[batch_mask].mean(axis=0)
            corrected_emb[batch_mask] -= batch_mean

    # 5. Store the corrected embedding
    adata_copy.obsm["X_emb"] = corrected_emb

    return adata_copy