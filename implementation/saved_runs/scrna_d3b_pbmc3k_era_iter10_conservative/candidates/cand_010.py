import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and Log-transform
    # These steps are standard for scRNA-seq data and are part of the
    # "STRONG, RELIABLE DIRECTION" identified in the problem description.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure adata.X is dense for subsequent numpy operations and PCA.
    # adata.X might be sparse (e.g., csr_matrix) after scanpy preprocessing.
    X_processed = adata.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed, dtype=np.float32)

    # Get batch information
    batches = adata.obs["batch"].values
    unique_batches = np.unique(batches)

    # 2. Batch effect correction in EXPRESSION space (per-gene mean subtraction per batch)
    # This addresses point 5 from the helpful building blocks:
    # "subtract each BATCH's MEAN per gene in EXPRESSION space, BEFORE PCA".
    # This step aims to remove batch-specific shifts in gene expression profiles.
    X_corrected_expr_space = np.copy(X_processed)

    for batch in unique_batches:
        batch_mask = (batches == batch)
        if np.any(batch_mask):
            # Calculate the mean expression for each gene within the current batch
            batch_mean_expression = np.mean(X_processed[batch_mask], axis=0)
            # Subtract this batch-specific mean from all cells belonging to this batch.
            # This centers the gene expression of each batch around zero for each gene.
            X_corrected_expr_space[batch_mask] -= batch_mean_expression

    # 3. Dimensionality reduction using PCA
    # The PCA is performed on the expression data that has been corrected for batch
    # effects by per-gene mean subtraction.
    n_samples, n_features = X_corrected_expr_space.shape
    
    # Handle edge case where there's only one sample; PCA is not meaningful for variance.
    if n_samples <= 1:
        # Return a trivial 1-dimensional embedding, fulfilling the shape requirement.
        adata.obsm["X_emb"] = np.zeros((n_samples, 1), dtype=np.float32)
        return adata

    # Determine the number of components for PCA.
    # 20 components is a common choice, but ensure it's valid for the data dimensions.
    # PCA can compute at most min(n_samples-1, n_features) components.
    n_comps = min(20, n_samples - 1, n_features) 
    n_comps = max(1, n_comps) # Ensure at least 1 component is always requested.

    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X_corrected_expr_space) # Compute PCA embedding

    # 4. Batch effect correction in EMBEDDING space (per-batch mean subtraction)
    # This addresses point 4 from the helpful building blocks:
    # "subtract each BATCH's MEAN vector in EMBEDDING space, AFTER PCA".
    # This is highlighted in the problem description as the "BEST simple method"
    # for batch mixing while preserving biological variation.
    emb_corrected = np.copy(emb)

    for batch in unique_batches:
        batch_mask = (batches == batch)
        if np.any(batch_mask):
            # Calculate the mean of embeddings for the current batch
            batch_mean_embedding = np.mean(emb[batch_mask], axis=0)
            # Subtract this batch-specific mean from all embeddings belonging to this batch.
            # This centers each batch's cells in the embedding space.
            emb_corrected[batch_mask] -= batch_mean_embedding

    # Store the final batch-corrected embedding in adata.obsm
    adata.obsm["X_emb"] = emb_corrected

    return adata