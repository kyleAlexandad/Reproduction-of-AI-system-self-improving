import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
import sys # For float_info.epsilon for robustness

def eliminate_batch_effect_fn(adata, config):
    """
    Performs batch integration on single-cell RNA-seq data by applying a two-stage process:
    1. Standard preprocessing (normalization, log-transformation).
    2. Crucially, per-batch gene standardization (z-scoring) to remove batch-specific
       mean and variance effects at the gene level. This aligns gene expression
       distributions across batches directly in the high-dimensional space.
    3. Final dimensionality reduction using PCA on the batch-standardized data
       to produce a low-dimensional embedding that mixes batches and preserves
       biological structure.

    Args:
        adata: An AnnData object containing raw gene-expression counts in adata.X and
               batch labels in adata.obs["batch"].
        config: A dictionary for potential configuration parameters (not used in this solution).

    Returns:
        An AnnData object with a new low-dimensional embedding stored in adata.obsm["X_emb"].
        The embedding aims to mix batches while preserving biological signals.
    """
    adata = adata.copy()

    # 1. Standard Preprocessing: Normalize total counts per cell and then log-transform.
    # This stabilizes variance and makes data more amenable to linear models and PCA.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure adata.X is a dense NumPy array for scikit-learn compatibility and performance.
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()

    # 2. Per-Batch Gene Standardization (Key Improvement for batch effect removal).
    # For each batch, we standardize the gene expression values (z-score per gene).
    # This removes batch-specific mean and variance differences for each gene,
    # effectively aligning gene expression distributions across batches *before* PCA.
    
    # Initialize an array to store the full batch-standardized data.
    X_corrected_full = np.empty_like(adata.X, dtype=float)
    
    for batch_label in adata.obs["batch"].unique():
        # Select cells belonging to the current batch
        batch_mask = (adata.obs["batch"] == batch_label)
        X_batch = adata[batch_mask, :].X
        
        # Calculate mean and standard deviation for each gene within this batch.
        # These statistics will be used to standardize only the cells within this batch.
        mean_genes_batch = X_batch.mean(axis=0)
        std_genes_batch = X_batch.std(axis=0)
        
        # Handle genes with zero standard deviation within the batch to prevent division by zero.
        # If a gene has zero variance, it means all cells in that batch have the same expression.
        # After centering, it should become 0. We replace 0 std with a small epsilon
        # or 1.0 (if all values are identical, mean=X, so X-mean is 0 anyway).
        # Using a small epsilon is generally more robust for numerical stability.
        std_genes_batch[std_genes_batch == 0] = sys.float_info.epsilon
        
        # Z-score standardization for genes within this specific batch.
        # This removes the batch-specific mean and scales by the batch-specific standard deviation.
        X_batch_standardized = (X_batch - mean_genes_batch) / std_genes_batch
        
        # Place the standardized data back into the full corrected array at the original cell positions.
        X_corrected_full[batch_mask, :] = X_batch_standardized
        
    # Update adata.X with the batch-standardized gene expression data.
    adata.X = X_corrected_full

    # 3. Final Dimensionality Reduction using PCA.
    # After per-batch standardization, PCA will now primarily capture the main biological variations
    # that are not confounded by batch-specific mean/variance differences, as those have been removed.
    
    # The number of components is chosen to be moderate (e.g., 50),
    # but also capped by the actual dimensions of the data to prevent errors.
    initial_n_comps = int(min(50, adata.X.shape[1] - 1, adata.X.shape[0] - 1))
    if initial_n_comps < 1:
        initial_n_comps = 1  # Ensure at least one component is computed

    # Apply PCA to the batch-standardized data.
    # `random_state` ensures reproducibility of the PCA results.
    pca = PCA(n_components=initial_n_comps, random_state=0)
    X_emb = pca.fit_transform(adata.X)
    
    # 4. Store the final batch-corrected low-dimensional embedding in adata.obsm["X_emb"].
    adata.obsm["X_emb"] = X_emb

    return adata