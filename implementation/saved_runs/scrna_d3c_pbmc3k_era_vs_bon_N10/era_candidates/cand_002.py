import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # For a robust scaling alternative to sc.pp.scale if needed, but sc.pp.scale is preferred.


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and log-transform the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 2. Add-on: Scale genes to unit variance and clip values.
    # This helps prevent highly variable genes from dominating PCA.
    # sc.pp.scale applies global centering and scaling.
    sc.pp.scale(adata, max_value=10)

    # Prepare a dense numpy array for further processing
    # adata.X might be sparse after previous steps, ensure it's dense for subsequent numpy ops
    X_processed = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X.copy()
    X_processed = np.asarray(X_processed, dtype=np.float32)

    # Get batch labels for per-batch operations
    batch_labels = adata.obs["batch"].values
    unique_batches = np.unique(batch_labels)

    # 3. Add-on: Subtract each batch's mean per gene in expression space, BEFORE PCA.
    # This removes batch-specific expression shifts directly from the gene features.
    for batch in unique_batches:
        batch_mask = (batch_labels == batch)
        # Calculate the mean expression for each gene within this batch
        batch_mean_expression = X_processed[batch_mask].mean(axis=0)
        # Subtract this batch-specific mean from all cells in this batch
        X_processed[batch_mask] -= batch_mean_expression

    # 4. Perform PCA to get an initial low-dimensional embedding
    # Determine n_components safely: at most 20, but not more than (n_samples-1) or (n_features-1).
    # PCA requires n_components <= min(n_samples, n_features).
    # We ensure n_comps is at least 1.
    n_samples, n_features = X_processed.shape
    n_comps = min(20, n_features - 1, n_samples - 1)
    if n_comps < 1:
        n_comps = 1 # Ensure at least one component, handles very small datasets gracefully

    pca = PCA(n_components=n_comps, random_state=0)
    initial_emb = pca.fit_transform(X_processed)

    # 5. Batch effect correction: Subtract batch centroids in the embedding space
    # This step aligns the batches in the low-dimensional embedding.
    corrected_emb = np.copy(initial_emb)

    for batch in unique_batches:
        # Identify cells belonging to the current batch
        batch_mask = (batch_labels == batch)

        # Calculate the mean embedding (centroid) for cells in this batch
        batch_centroid = initial_emb[batch_mask].mean(axis=0)

        # Subtract the batch centroid from all cells in this batch
        corrected_emb[batch_mask] -= batch_centroid

    # 6. Store the batch-corrected embedding in adata.obsm["X_emb"]
    adata.obsm["X_emb"] = corrected_emb

    return adata