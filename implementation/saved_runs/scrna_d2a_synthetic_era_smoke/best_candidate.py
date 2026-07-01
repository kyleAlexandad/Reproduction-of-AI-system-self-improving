import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    # Create a copy to avoid modifying the input AnnData object directly
    adata = adata.copy()

    # 1. Preprocessing: Normalize total counts per cell and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Get the expression matrix and ensure it's a dense NumPy array for PCA
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    # 2. Initial Dimensionality Reduction (PCA)
    # Determine the number of components for PCA.
    # Max_desired_comps is a reasonable upper bound; actual components will be capped
    # by the smaller of n_samples and n_features, handled gracefully by sklearn's PCA.
    max_desired_comps = 30
    n_comps = min(max_desired_comps, adata.n_obs, adata.n_vars)

    # Handle edge case where PCA cannot be performed (e.g., empty data or insufficient dimensions).
    # PCA requires n_components >= 1 to return a meaningful embedding.
    if n_comps == 0 or adata.n_obs == 0 or adata.n_vars == 0:
        # If no valid components or no data, return an empty (n_cells, 1) or (0, 1) array.
        # This satisfies the requirement for a 2D float array.
        adata.obsm["X_emb"] = np.empty((adata.n_obs, 1), dtype=np.float32)
        return adata
    
    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X)

    # 3. Batch Effect Correction in PCA space (batch-centering)
    # This step aims to align the centroids of different batches in the embedding space,
    # thereby removing batch-specific shifts while preserving within-batch biological variance.
    batch_labels = adata.obs["batch"]
    if batch_labels.dtype.name != 'category':
        batch_labels = batch_labels.astype('category')
    
    # Calculate the global mean of the embedding.
    # This mean will be added back to each batch after subtracting its specific mean,
    # effectively aligning all batch centroids to the overall dataset centroid.
    global_mean_emb = np.mean(emb, axis=0)

    # Iterate through each unique batch and apply the mean correction
    for batch_id in batch_labels.cat.categories:
        batch_indices = (batch_labels == batch_id).to_numpy() # Boolean mask for cells in current batch
        
        # Ensure there are cells belonging to the current batch to avoid errors
        if np.any(batch_indices):
            batch_mean_emb = np.mean(emb[batch_indices], axis=0)
            # Subtract batch mean and add global mean to correct for batch-specific shifts
            emb[batch_indices] = emb[batch_indices] - batch_mean_emb + global_mean_emb

    # 4. Store the corrected low-dimensional embedding
    adata.obsm["X_emb"] = emb

    return adata