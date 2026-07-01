import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and log-transform the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure adata.X is a dense numpy array of float32 for subsequent gene-level
    # operations and PCA, which are more efficient with dense arrays.
    X_dense = adata.X
    if hasattr(X_dense, "toarray"):
        X_dense = X_dense.toarray()
    adata.X = np.asarray(X_dense, dtype=np.float32) # Store dense float32 array back to adata.X

    # Get batch labels for corrections
    batch_labels = adata.obs["batch"].values
    unique_batches = np.unique(batch_labels)

    # 2. NEW: Subtract each BATCH's MEAN per gene in EXPRESSION space
    # This step aims to remove batch-specific mean shifts at the gene level before
    # dimensionality reduction, allowing PCA to capture more biological variance.
    for batch in unique_batches:
        batch_mask = (batch_labels == batch)
        # Calculate the mean expression for each gene within this batch
        batch_gene_mean = adata.X[batch_mask].mean(axis=0)
        # Subtract this mean from all cells belonging to this batch
        adata.X[batch_mask] -= batch_gene_mean

    # 3. NEW: Robust scaling of genes
    # This standardizes gene expression values by centering to zero mean and scaling
    # to unit variance, capping extreme values to max_value=10. This helps in
    # making PCA less sensitive to outliers and genes with very different scales.
    sc.pp.scale(adata, max_value=10)

    # 4. Perform PCA to get an initial low-dimensional embedding
    # Determine n_components safely: at most 20, but not more than (n_samples-1)
    # or (n_features-1). PCA requires n_components <= min(n_samples, n_features) - 1.
    n_comps = min(20, adata.X.shape[1] - 1, adata.X.shape[0] - 1)
    if n_comps < 1:
        n_comps = 1 # Ensure at least one component, handles very small datasets gracefully

    pca = PCA(n_components=n_comps, random_state=0)
    # adata.X is now dense, gene-level batch corrected, and scaled; ready for PCA.
    initial_emb = pca.fit_transform(adata.X)

    # 5. Batch effect correction: Subtract batch centroids in the embedding space
    # This is a highly effective and simple method for batch mixing, by centering
    # each batch's projection in the PCA space.
    corrected_emb = np.copy(initial_emb)
    for batch in unique_batches:
        batch_mask = (batch_labels == batch)
        # Calculate the mean embedding (centroid) for cells in this batch
        batch_centroid = initial_emb[batch_mask].mean(axis=0)
        # Subtract the batch centroid from all cells in this batch
        corrected_emb[batch_mask] -= batch_centroid

    # 6. Store the final batch-corrected embedding in adata.obsm["X_emb"]
    adata.obsm["X_emb"] = corrected_emb

    return adata