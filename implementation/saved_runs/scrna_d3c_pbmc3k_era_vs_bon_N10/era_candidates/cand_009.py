import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata_copy = adata.copy()

    # 1. Preprocessing: Normalize and log-transform the data
    sc.pp.normalize_total(adata_copy, target_sum=1e4)
    sc.pp.log1p(adata_copy)

    # Ensure adata.X is a dense numpy array for robust processing.
    # scanpy's pp.normalize_total and pp.log1p usually convert to dense float32,
    # but explicit conversion adds robustness, especially if input was sparse.
    X_lognorm = adata_copy.X
    if hasattr(X_lognorm, "toarray"):
        X_lognorm = X_lognorm.toarray()
    X_lognorm = np.asarray(X_lognorm, dtype=np.float32)

    # 2. Batch effect correction in EXPRESSION space (before PCA).
    # This step implements "subtract each BATCH's MEAN per gene in EXPRESSION space".
    # This directly addresses nuisance shifts at the gene level.
    X_corrected_expr = np.copy(X_lognorm)
    batch_labels = adata_copy.obs["batch"].values
    unique_batches = np.unique(batch_labels)

    for batch in unique_batches:
        batch_mask = (batch_labels == batch)
        # Calculate the mean expression for each gene within this batch
        batch_gene_means = X_corrected_expr[batch_mask].mean(axis=0)
        # Subtract these gene-specific batch means from all cells in this batch
        X_corrected_expr[batch_mask] -= batch_gene_means

    # 3. Perform PCA to get an initial low-dimensional embedding.
    # PCA is applied to the expression-space batch-corrected data.
    # Determine n_components safely: up to 50 (a common choice for biological datasets),
    # but not more than (n_samples-1) or (n_features-1).
    n_samples, n_features = X_corrected_expr.shape
    n_comps = min(50, n_features - 1, n_samples - 1)
    if n_comps < 1:
        n_comps = 1 # Ensure at least one component if data is too small

    pca = PCA(n_components=n_comps, random_state=0)
    initial_emb = pca.fit_transform(X_corrected_expr)

    # 4. Batch effect correction in EMBEDDING space (after PCA).
    # This step implements "subtract each BATCH's MEAN vector in EMBEDDING space",
    # which is highlighted as the "BEST simple method" in the prompt.
    corrected_emb = np.copy(initial_emb)
    # The batch_labels and unique_batches from step 2 can be reused here.

    for batch in unique_batches:
        batch_mask = (batch_labels == batch)
        # Calculate the mean embedding (centroid) for cells in this batch.
        batch_centroid = initial_emb[batch_mask].mean(axis=0)
        # Subtract the batch centroid from all cells in this batch.
        corrected_emb[batch_mask] -= batch_centroid

    # 5. Store the final batch-corrected embedding in adata.obsm["X_emb"].
    adata_copy.obsm["X_emb"] = corrected_emb

    return adata_copy