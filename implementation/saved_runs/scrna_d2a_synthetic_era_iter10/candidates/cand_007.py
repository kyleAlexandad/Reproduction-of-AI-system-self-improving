import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    """
    Performs batch integration on single-cell RNA-seq data by applying per-gene
    batch mean centering followed by Principal Component Analysis (PCA).

    This approach first normalizes and log-transforms the gene expression data.
    Then, for each gene, it removes batch-specific mean differences by subtracting
    the batch mean and adding back the global mean. This ensures that batch effects
    are removed at the gene level before dimensionality reduction.
    Finally, PCA is applied to the batch-corrected gene expression matrix to produce
    a low-dimensional embedding that preserves biological variation while mixing batches.

    Args:
        adata: An AnnData object containing raw gene-expression counts in adata.X and
               batch labels in adata.obs["batch"].
        config: A dictionary for potential configuration parameters (not used in this solution).

    Returns:
        An AnnData object with a new low-dimensional embedding stored in adata.obsm["X_emb"].
        The embedding will have batch effects reduced and biological structure preserved.
    """
    # Create a copy to avoid modifying the original AnnData object in place.
    adata = adata.copy()

    # 1. Preprocessing: Normalize total counts per cell and then log-transform.
    # This standard step helps to stabilize variance and make gene expression distributions
    # more amenable to linear models and dimensionality reduction.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure adata.X is a dense NumPy array. This is important for scikit-learn
    # compatibility and performance with subsequent operations.
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    
    # Get the preprocessed expression data and batch labels
    X = adata.X
    batch_labels = adata.obs["batch"]

    # 2. Batch Effect Correction: Per-gene mean centering.
    # For each gene, we calculate the mean expression within each batch and subtract it.
    # To prevent shifting the overall mean of the dataset, the global mean for each gene
    # is then added back. This effectively aligns the distributions of each gene across batches
    # without distorting the overall gene expression levels.
    
    # Calculate the global mean for each gene across all cells
    global_means = X.mean(axis=0)
    
    # Initialize an array for the batch-corrected data
    X_corrected = X.copy()
    
    # Iterate through each unique batch present in the dataset
    for batch_id in batch_labels.unique():
        # Create a boolean mask to select cells belonging to the current batch
        batch_mask = (batch_labels == batch_id)
        
        # Extract data for the current batch
        batch_data = X[batch_mask, :]
        
        # Calculate the mean expression for each gene within this batch
        batch_means = batch_data.mean(axis=0)
        
        # Subtract the batch mean and add back the global mean for cells in this batch.
        # This shifts the batch-specific gene means to the global gene means.
        X_corrected[batch_mask, :] = X_corrected[batch_mask, :] - batch_means + global_means

    # Replace the original data with the batch-corrected data in the AnnData object.
    # Subsequent operations like PCA will now operate on the corrected data.
    adata.X = X_corrected

    # 3. Dimensionality Reduction using PCA on the batch-corrected data.
    # PCA is applied to extract the main components of variation, which should now
    # primarily reflect biological differences due to the prior batch correction.
    # We choose a moderate number of components, capped by data dimensions,
    # to capture significant biological signal while further reducing noise.
    initial_n_comps = int(min(100, adata.X.shape[1] - 1, adata.X.shape[0] - 1))
    if initial_n_comps < 1:
        initial_n_comps = 1  # Ensure at least one component is computed

    # `sc.pp.pca` computes PCA and stores the result in `adata.obsm["X_pca"]`.
    # `random_state` ensures reproducibility of PCA results.
    sc.pp.pca(adata, n_comps=initial_n_comps, random_state=0)
    
    # 4. Store the final batch-corrected low-dimensional embedding.
    # The output of PCA (stored in `adata.obsm["X_pca"]`) is the desired embedding.
    adata.obsm["X_emb"] = adata.obsm["X_pca"]

    # 5. Return the AnnData object with the new embedding.
    return adata