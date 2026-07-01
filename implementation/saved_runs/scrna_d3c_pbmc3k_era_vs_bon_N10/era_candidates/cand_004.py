import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and log-transform the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure adata.X is a dense numpy array of float32 for subsequent
    # numerical stability and compatibility with PCA and array operations.
    X_processed = adata.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed, dtype=np.float32)

    batch_labels = adata.obs["batch"].values
    unique_batches = np.unique(batch_labels)

    # 2. Batch effect correction in EXPRESSION space (per-gene, per-batch mean subtraction)
    # This step aims to remove batch-specific mean shifts for each gene
    # before dimensionality reduction.
    for batch in unique_batches:
        batch_mask = (batch_labels == batch)
        
        # Calculate the mean expression for each gene within the current batch
        # This results in a vector of means, one for each gene.
        batch_mean_per_gene = X_processed[batch_mask].mean(axis=0)
        
        # Subtract this batch-specific gene mean from all cells in the current batch.
        # After this, the mean expression of each gene *within each batch* will be approximately zero.
        X_processed[batch_mask] -= batch_mean_per_gene

    # 3. Perform PCA on the batch-corrected expression data
    # Determine n_components safely: at most 20, but not more than (n_samples-1) or (n_features-1).
    # PCA requires n_components <= min(n_samples, n_features). Using -1 for safety and to match parent.
    n_comps = min(20, X_processed.shape[1] - 1, X_processed.shape[0] - 1)
    if n_comps < 1:
        n_comps = 1 # Ensure at least one component, handles very small datasets gracefully

    pca = PCA(n_components=n_comps, random_state=0)
    initial_emb = pca.fit_transform(X_processed)

    # 4. Batch effect correction in EMBEDDING space (subtract batch centroids)
    # This further refines batch mixing by centering each batch's embedding
    # around the origin in the low-dimensional space.
    corrected_emb = np.copy(initial_emb)
    for batch in unique_batches:
        batch_mask = (batch_labels == batch)
        
        # Calculate the mean embedding (centroid) for cells in this batch
        batch_centroid = initial_emb[batch_mask].mean(axis=0)
        
        # Subtract the batch centroid from all cells in this batch.
        # This makes the mean embedding of each batch approximately zero.
        corrected_emb[batch_mask] -= batch_centroid

    # 5. Store the final batch-corrected embedding in adata.obsm["X_emb"]
    adata.obsm["X_emb"] = corrected_emb

    return adata