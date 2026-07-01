import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    # Create a copy to avoid modifying the input AnnData object directly
    adata = adata.copy()

    # 1. Preprocessing: Normalize total counts, log-transform, and select highly variable genes
    # Normalization and log-transformation are standard steps for scRNA-seq data.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Select Highly Variable Genes (HVGs) to focus on biological signal and reduce noise.
    # This also significantly reduces the dimensionality for subsequent steps, improving speed.
    # Parameters for HVG selection can be critical; these are robust defaults.
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=2000)
    
    # Filter the AnnData object to include only the selected HVGs.
    # Use .copy() to ensure a new AnnData object is created, preventing view/copy issues.
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Get the expression matrix for HVGs and ensure it's a dense NumPy array (float32).
    # This prepares the data for efficient numerical operations.
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # Handle edge case: if no cells or no genes remain after filtering
    if X.shape[0] == 0 or X.shape[1] == 0:
        # Return an empty (n_cells, 1) or (0, 1) array, satisfying the shape requirement.
        adata.obsm["X_emb"] = np.empty((adata.n_obs, 1), dtype=np.float32)
        return adata

    # 2. Per-gene Batch Effect Correction (before PCA)
    # This strategy performs gene-wise mean centering for each batch, effectively removing
    # batch-specific shifts in gene expression means while attempting to preserve within-batch
    # biological variance. This is a robust and lightweight alternative to post-PCA correction.
    
    batch_labels = adata.obs["batch"]
    # Ensure batch labels are categorical for robust iteration over categories.
    if batch_labels.dtype.name != 'category':
        batch_labels = batch_labels.astype('category')

    # Calculate the global mean for each gene across all cells.
    global_means_per_gene = np.mean(X, axis=0)

    # Create a copy of the expression matrix to store the corrected values.
    X_corrected = X.copy()

    # Iterate through each unique batch to apply the gene-wise mean correction.
    for batch_id in batch_labels.cat.categories:
        # Identify cells belonging to the current batch.
        batch_indices = (batch_labels == batch_id).to_numpy()

        # Apply correction only if there are cells in the current batch.
        if np.any(batch_indices):
            # Extract expression data for the current batch.
            batch_X = X[batch_indices]

            # Calculate the mean for each gene within this specific batch.
            batch_means_per_gene = np.mean(batch_X, axis=0)

            # Apply the correction: for each cell in the batch, subtract the batch-specific
            # gene mean and add back the global gene mean. This aligns batch means to the global mean.
            X_corrected[batch_indices] = batch_X - batch_means_per_gene + global_means_per_gene

    # 3. Dimensionality Reduction (PCA) on the batch-corrected data
    # Use a higher number of components than the parent to potentially capture more biological
    # variance, as the batch effects are presumed to be largely removed prior to this step.
    max_desired_comps = 50
    n_comps = min(max_desired_comps, X_corrected.shape[0], X_corrected.shape[1])

    # Handle edge case where PCA cannot be performed (e.g., insufficient dimensions after correction).
    if n_comps < 1:
        # If no valid components, return an (n_cells, 1) array to meet the requirement.
        adata.obsm["X_emb"] = np.empty((adata.n_obs, 1), dtype=np.float32)
        return adata

    # Initialize and fit PCA on the batch-corrected expression matrix.
    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X_corrected)

    # 4. Store the final low-dimensional embedding in adata.obsm["X_emb"]
    adata.obsm["X_emb"] = emb

    return adata