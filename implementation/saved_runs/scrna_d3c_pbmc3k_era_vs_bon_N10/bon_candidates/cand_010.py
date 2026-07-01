import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    adata_processed = adata.copy()

    # 1. Normalize and log-transform the raw counts
    sc.pp.normalize_total(adata_processed, target_sum=1e4)
    sc.pp.log1p(adata_processed)

    # 2. Scale features (genes) to unit variance and clip outliers
    # This is a common preprocessing step that helps PCA by giving all features
    # a similar scale, making it robust to highly variable genes.
    sc.pp.scale(adata_processed, max_value=10)

    # Ensure the expression matrix is dense for PCA.
    # While scanpy's `pp.pca` can handle sparse matrices, `sklearn.decomposition.PCA`
    # (used by the parent candidate) generally works more efficiently with dense arrays.
    X_dense = adata_processed.X
    if hasattr(X_dense, "toarray"):
        X_dense = X_dense.toarray()
    X_dense = np.asarray(X_dense, dtype=np.float32)

    # 3. Determine the number of components for PCA.
    # Using 20 components as a reasonable default for typical scRNA-seq datasets,
    # ensuring it does not exceed the number of cells or genes minus one.
    n_cells, n_genes = X_dense.shape
    n_comps = int(min(20, n_genes - 1, n_cells - 1))
    # Ensure n_comps is at least 1, even in edge cases with very few cells/genes.
    n_comps = max(1, n_comps)

    # 4. Perform PCA to get an initial low-dimensional embedding.
    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X_dense)

    # 5. Subtract each BATCH's MEAN vector in the EMBEDDING space.
    # This step directly addresses the batch effect by centering each batch's
    # distribution in the embedding space, thus mixing them while preserving
    # within-batch biological variance.
    batch_labels = adata_processed.obs["batch"]
    unique_batches = batch_labels.unique()

    for batch in unique_batches:
        batch_mask = (batch_labels == batch)
        # Only perform correction if there are cells belonging to the current batch
        if np.any(batch_mask):
            batch_mean_emb = emb[batch_mask].mean(axis=0)
            emb[batch_mask] -= batch_mean_emb

    # 6. Store the batch-corrected embedding in adata.obsm["X_emb"].
    adata_processed.obsm["X_emb"] = emb

    return adata_processed