import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and log-transform the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure adata.X is a dense numpy array for robust processing.
    # scanpy's pp.normalize_total and pp.log1p usually convert to dense float32,
    # but explicit conversion adds robustness.
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32) # Ensure float32 for consistency

    # 2. Perform PCA to get an initial low-dimensional embedding.
    # This step follows the advice to use PCA directly after log-normalization.
    # Determine n_components safely: at most 20 (as per common practice and problem context),
    # but not more than (n_samples-1) or (n_features-1).
    # PCA requires n_components <= min(n_samples, n_features).
    # We ensure n_comps is at least 1 for very small datasets.
    n_samples, n_features = X.shape
    n_comps = min(20, n_features - 1, n_samples - 1)
    if n_comps < 1:
        n_comps = 1 # Ensure at least one component if data is too small

    pca = PCA(n_components=n_comps, random_state=0)
    initial_emb = pca.fit_transform(X)

    # 3. Batch effect correction in EMBEDDING space (after PCA).
    # This is the "BEST simple method" described in the prompt:
    # subtracting each batch's mean embedding vector (batch-centering in embedding space).
    corrected_emb = np.copy(initial_emb)
    batch_labels = adata.obs["batch"].values
    unique_batches = np.unique(batch_labels)

    for batch in unique_batches:
        batch_mask = (batch_labels == batch)
        # Calculate the mean embedding (centroid) for cells in this batch.
        batch_centroid = initial_emb[batch_mask].mean(axis=0)
        # Subtract the batch centroid from all cells in this batch.
        corrected_emb[batch_mask] -= batch_centroid

    # 4. Store the final batch-corrected embedding in adata.obsm["X_emb"].
    adata.obsm["X_emb"] = corrected_emb

    return adata