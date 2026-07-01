import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and Log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Get data matrix for further processing.
    # Convert to dense array for sklearn PCA and ensure float type.
    # This also creates a copy, allowing safe modification for gene-level batch correction.
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # 2. Batch effect correction in expression space (per-gene mean subtraction per batch)
    # This step subtracts each batch's mean expression for each gene.
    # This helps to remove linear batch effects directly from the gene expression data
    # before dimensionality reduction.
    batches = adata.obs["batch"].values
    unique_batches = np.unique(batches)

    # Perform in-place correction on the X array
    for batch in unique_batches:
        batch_mask = (batches == batch)
        if np.any(batch_mask):
            # Calculate the mean of gene expression for the current batch across all genes
            batch_mean_expression = np.mean(X[batch_mask, :], axis=0)
            # Subtract this batch-specific mean from all cells belonging to this batch
            X[batch_mask, :] -= batch_mean_expression
    
    # 3. Dimensionality reduction using PCA
    # Determine n_components for PCA, ensuring it's valid:
    # at least 1, and less than min(n_samples, n_features).
    # Limiting to 20 components is a common practice for scRNA-seq embeddings.
    n_comps = int(min(20, X.shape[1] - 1, X.shape[0] - 1))
    n_comps = max(1, n_comps) # Ensure at least 1 component

    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X) # PCA on the expression-space batch-corrected data

    # 4. Batch effect correction in the embedding space (mean subtraction per batch)
    # This step centers the cells of each batch around the origin in the PCA embedding space.
    # This is a highly effective method for batch mixing while preserving biological variation.
    emb_corrected = np.copy(emb) # Create a copy of the embedding to modify

    for batch in unique_batches:
        batch_mask = (batches == batch)
        if np.any(batch_mask):
            # Calculate the mean of embeddings for the current batch
            batch_mean_embedding = np.mean(emb[batch_mask], axis=0)
            # Subtract the batch mean from all embeddings belonging to this batch
            emb_corrected[batch_mask] -= batch_mean_embedding

    # Store the final batch-corrected embedding in adata.obsm
    adata.obsm["X_emb"] = emb_corrected

    return adata