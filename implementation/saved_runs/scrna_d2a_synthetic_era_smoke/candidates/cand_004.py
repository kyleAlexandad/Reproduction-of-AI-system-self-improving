import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    # Create a copy to avoid modifying the input AnnData object directly
    adata_copy = adata.copy()

    # --- Preprocessing ---
    # 1. Normalize total counts per cell to 1e4 and log-transform
    sc.pp.normalize_total(adata_copy, target_sum=1e4)
    sc.pp.log1p(adata_copy)

    # 2. Scale each gene to have mean 0 and variance 1, clipping extreme values
    # This standardizes gene expression values, making them comparable and
    # preventing highly expressed genes from dominating PCA.
    # max_value clips values after scaling, helping with outlier robustness.
    sc.pp.scale(adata_copy, max_value=10)

    # Ensure the expression matrix is a dense NumPy array for subsequent operations.
    # This is important for efficient mean calculations and batch correction,
    # as scanpy's internal .X might be sparse after scaling.
    X_processed = adata_copy.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed) # Ensure it's a NumPy array for consistency

    # --- Gene-level Batch Effect Correction (mean centering per gene within batches) ---
    # This step removes batch-specific mean shifts directly from the gene expression matrix.
    # By standardizing the data (sc.pp.scale), each gene already has a global mean of 0.
    # Therefore, we only need to subtract the batch-specific mean for each gene;
    # this effectively aligns all batch means to the global mean of 0 for each gene.
    
    batch_labels = adata_copy.obs["batch"]
    if batch_labels.dtype.name != 'category':
        batch_labels = batch_labels.astype('category')
    
    # Iterate through each unique batch and apply the mean correction per gene
    for batch_id in batch_labels.cat.categories:
        batch_indices = (batch_labels == batch_id).to_numpy() # Boolean mask for cells in current batch
        
        # Ensure there are cells belonging to the current batch to avoid errors
        if np.any(batch_indices):
            # Calculate the mean expression for each gene within the current batch
            batch_gene_means = np.mean(X_processed[batch_indices], axis=0)
            
            # Subtract the batch-specific mean from all cells in that batch for each gene.
            # Since the data is already scaled to have a global mean of 0,
            # adding the global mean back is implicitly handled (as it's 0).
            X_processed[batch_indices] = X_processed[batch_indices] - batch_gene_means

    # --- Initial Dimensionality Reduction (PCA) ---
    # Handle edge cases for empty or insufficient data before performing PCA.
    if adata_copy.n_obs == 0 or adata_copy.n_vars == 0:
        # If there's no data or no genes, return an empty (n_cells, 1) array as per requirements.
        adata_copy.obsm["X_emb"] = np.empty((adata_copy.n_obs, 1), dtype=np.float32)
        return adata_copy
    
    max_desired_comps = 50 # A reasonable upper bound for the number of PCA components.
    
    # Determine the actual number of components PCA can extract, bounded by
    # max_desired_comps, the number of samples, and the number of features.
    n_components_to_fit = min(max_desired_comps, adata_copy.n_obs, adata_copy.n_vars)
    
    # If, after considering all bounds, PCA cannot extract at least 1 component,
    # return a 1-D embedding (e.g., if all features are constant after processing).
    if n_components_to_fit < 1:
        adata_copy.obsm["X_emb"] = np.empty((adata_copy.n_obs, 1), dtype=np.float32)
        return adata_copy
    
    # Perform PCA on the batch-corrected gene expression matrix.
    pca = PCA(n_components=n_components_to_fit, random_state=0)
    emb = pca.fit_transform(X_processed)

    # --- Store the corrected low-dimensional embedding ---
    adata_copy.obsm["X_emb"] = emb

    return adata_copy