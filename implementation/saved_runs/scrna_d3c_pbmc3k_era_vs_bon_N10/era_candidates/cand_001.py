import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and log-transform the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure adata.X is a dense numpy array for PCA
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32) # Ensure float32 for consistency and memory if needed

    # 2. Perform PCA to get an initial low-dimensional embedding
    # Determine n_components safely: at most 20, but not more than (n_samples-1) or (n_features-1).
    # PCA requires n_components <= min(n_samples, n_features) - 1.
    # We ensure n_comps is at least 1.
    n_comps = min(20, X.shape[1] - 1, X.shape[0] - 1)
    if n_comps < 1:
        n_comps = 1 # Ensure at least one component, handles very small datasets gracefully

    pca = PCA(n_components=n_comps, random_state=0)
    initial_emb = pca.fit_transform(X)

    # 3. Batch effect correction: Subtract batch centroids in the embedding space
    batch_labels = adata.obs["batch"].values
    unique_batches = np.unique(batch_labels)

    # Create a copy of the embedding to modify
    corrected_emb = np.copy(initial_emb)

    for batch in unique_batches:
        # Identify cells belonging to the current batch
        batch_mask = (batch_labels == batch)

        # Calculate the mean embedding (centroid) for cells in this batch
        batch_centroid = initial_emb[batch_mask].mean(axis=0)

        # Subtract the batch centroid from all cells in this batch
        # This centers each batch's embedding around the origin (relative to its initial position)
        corrected_emb[batch_mask] -= batch_centroid

    # 4. Store the batch-corrected embedding in adata.obsm["X_emb"]
    adata.obsm["X_emb"] = corrected_emb

    return adata