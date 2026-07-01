import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    # Create a copy to avoid modifying the input AnnData object directly
    adata = adata.copy()

    # 1. Preprocessing: Normalize total counts per cell and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Convert the expression matrix to a dense NumPy array for consistency and performance
    # with subsequent numerical operations (gene-wise correction, PCA).
    X_processed = adata.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed, dtype=np.float32) # Ensure float32 type

    batch_labels = adata.obs["batch"]
    if batch_labels.dtype.name != 'category':
        batch_labels = batch_labels.astype('category')

    # Handle edge case: if no cells or no genes, return an empty embedding.
    if adata.n_obs == 0 or adata.n_vars == 0:
        adata.obsm["X_emb"] = np.empty((adata.n_obs, 1), dtype=np.float32)
        return adata

    # 2. Gene-wise Batch Effect Correction (per-batch mean correction of genes before PCA)
    # This step aligns the mean expression of each gene across different batches.
    # It subtracts the batch-specific mean for each gene and adds back the global gene mean,
    # effectively removing batch-specific shifts in gene expression.
    X_corrected = np.copy(X_processed) # Operate on a copy of the preprocessed data

    # Calculate the global mean expression for each gene across all cells
    global_gene_means = np.mean(X_processed, axis=0)

    # Iterate through each unique batch to apply the correction
    for batch_id in batch_labels.cat.categories:
        batch_indices = (batch_labels == batch_id).to_numpy() # Boolean mask for cells in current batch

        # Ensure there are cells belonging to the current batch to avoid errors
        if np.any(batch_indices):
            # Calculate mean expression for each gene within the current batch
            batch_gene_means = np.mean(X_processed[batch_indices, :], axis=0)

            # Apply correction: for cells in this batch, subtract batch mean and add global mean
            X_corrected[batch_indices, :] = X_processed[batch_indices, :] - batch_gene_means + global_gene_means

    # 3. Dimensionality Reduction (PCA) on the batch-corrected gene expression matrix
    max_desired_comps = 30 # A reasonable upper bound for synthetic data embedding dimensions

    # Filter out genes that have zero variance after batch correction.
    # These genes carry no information and can cause issues for PCA if n_components > n_features.
    gene_variances = np.var(X_corrected, axis=0)
    # Using a small threshold for numerical stability to catch near-zero variances
    non_zero_var_genes_idx = np.where(gene_variances > 1e-9)[0]

    if len(non_zero_var_genes_idx) == 0:
        # If all genes have zero variance (e.g., all constant after correction),
        # PCA cannot proceed, return an empty (n_cells, 1) embedding.
        adata.obsm["X_emb"] = np.empty((adata.n_obs, 1), dtype=np.float32)
        return adata
    
    # Select only the genes with non-zero variance
    X_corrected_filtered = X_corrected[:, non_zero_var_genes_idx]
    
    # Determine the number of PCA components, capped by max_desired_comps,
    # the number of cells, and the number of remaining (non-zero variance) genes.
    n_comps = min(max_desired_comps, adata.n_obs, X_corrected_filtered.shape[1])

    # Handle edge case where n_comps becomes zero after filtering (e.g., very few genes left)
    if n_comps == 0:
        adata.obsm["X_emb"] = np.empty((adata.n_obs, 1), dtype=np.float32)
        return adata

    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X_corrected_filtered)

    # 4. Store the corrected low-dimensional embedding
    adata.obsm["X_emb"] = emb

    return adata