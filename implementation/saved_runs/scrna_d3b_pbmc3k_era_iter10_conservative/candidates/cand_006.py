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

    # Prepare data for PCA.
    # PCA from scikit-learn expects a dense numpy array.
    # adata.X might be sparse (e.g., csr_matrix) after scanpy preprocessing.
    X_processed = adata.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed, dtype=np.float32)

    # 2. Dimensionality reduction using PCA
    # Determine the number of components for PCA.
    # We use a practical limit (e.g., 20 components) as is common in single-cell analysis
    # for initial embeddings. This number is also robust for the PBMC3k dataset size.
    # Ensure n_components is valid: at least 1, and does not exceed min(n_samples, n_features).
    n_samples, n_features = X_processed.shape
    
    # Handle the rare edge case where there's only one sample, PCA is not meaningful for variance reduction.
    if n_samples <= 1:
        # Return a trivial 1-dimensional embedding, fulfilling the shape requirement.
        # This case is highly unlikely for real scRNA-seq datasets like PBMC3k.
        adata.obsm["X_emb"] = np.zeros((n_samples, 1), dtype=np.float32)
        return adata

    # Calculate n_comps, ensuring it's positive and does not exceed data dimensions.
    # PCA can compute at most min(n_samples-1, n_features) components.
    n_comps = min(20, n_samples - 1, n_features) 
    n_comps = max(1, n_comps) # Ensure at least 1 component is always requested.

    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X_processed) # Compute PCA embedding

    # 3. Batch effect correction in the embedding space (mean subtraction per batch)
    # This step centers the cells of each batch around the origin in the PCA embedding space.
    # This is highlighted in the problem description as the "BEST simple method"
    # for batch mixing while preserving biological variation.
    batches = adata.obs["batch"].values
    unique_batches = np.unique(batches)

    emb_corrected = np.copy(emb) # Create a copy of the embedding to modify

    for batch in unique_batches:
        batch_mask = (batches == batch)
        if np.any(batch_mask):
            # Calculate the mean of embeddings for the current batch
            batch_mean_embedding = np.mean(emb[batch_mask], axis=0)
            # Subtract this batch-specific mean from all embeddings belonging to this batch
            emb_corrected[batch_mask] -= batch_mean_embedding

    # Store the final batch-corrected embedding in adata.obsm
    adata.obsm["X_emb"] = emb_corrected

    return adata