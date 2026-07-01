import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from scipy.sparse import issparse


def eliminate_batch_effect_fn(adata, config):
    """
    Performs batch integration on single-cell RNA-seq data by applying gene-level
    batch mean correction, followed by scaling and PCA.

    This method aims to remove batch effects at the gene expression level before
    dimensionality reduction, ensuring that PCA captures biological variance
    more effectively.

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
    # This is a standard first step to stabilize variance and make gene expression
    # distributions more amenable to linear methods.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure adata.X is a dense NumPy array for efficient matrix operations.
    # PCA and subsequent numpy operations perform best on dense arrays.
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)  # Ensure float32 for consistency and memory

    # 2. Gene-level batch correction: Subtract batch-specific means for each gene
    #    and then add back the global mean for that gene.
    # This step directly targets and removes the additive batch effect from each gene's
    # expression profile while preserving the overall mean expression for each gene.
    
    # Calculate global mean expression for each gene across all cells (before any correction)
    global_means_orig = X.mean(axis=0)
    
    # Initialize the matrix for corrected expression
    X_corrected = X.copy()
    
    # Get batch labels for iterating
    batch_labels = adata.obs["batch"]
    
    # Iterate through each unique batch
    for batch_id in batch_labels.unique():
        batch_mask = (batch_labels == batch_id)
        
        # Ensure there are cells for the current batch to avoid errors with empty slices.
        if np.sum(batch_mask) > 0:
            # Calculate the mean expression for each gene within the current batch
            batch_mean = X[batch_mask, :].mean(axis=0)
            
            # Subtract the batch mean from all cells belonging to this batch
            # This centers the expression of each gene for cells within this batch around zero
            X_corrected[batch_mask, :] -= batch_mean

    # Add back the original global means for each gene.
    # This ensures that while batch-specific deviations are removed, the overall
    # expression level of each gene across the entire dataset is preserved.
    X_corrected += global_means_orig

    # 3. Z-score scaling of genes after batch correction.
    # This standardizes each gene's expression to have zero mean and unit variance,
    # which is generally beneficial for PCA as it prevents highly expressed genes
    # from dominating the principal components. Values are clipped to prevent outliers.
    
    # Calculate means and standard deviations for scaling
    # Use ddof=1 for sample standard deviation, which is a common practice.
    means_scaled = X_corrected.mean(axis=0)
    stds_scaled = X_corrected.std(axis=0, ddof=1)
    
    # Replace zero standard deviations with 1 to avoid division by zero.
    # Genes with no variance will remain centered at 0 after this.
    stds_scaled[stds_scaled == 0] = 1 
    
    # Apply z-score scaling
    X_scaled = (X_corrected - means_scaled) / stds_scaled
    
    # Clip values to a reasonable range (e.g., -10 to 10) to mitigate the influence
    # of extreme outliers that might still be present after log-transformation.
    max_value_clip = 10
    X_scaled = np.clip(X_scaled, -max_value_clip, max_value_clip)
    
    # 4. Apply PCA to the batch-corrected and scaled data.
    # PCA reduces dimensionality, capturing the main axes of variation which,
    # after the preceding batch correction, should primarily represent biological signals.
    
    # Determine the number of principal components.
    # A common choice is 50, but we cap it to ensure it's less than
    # the number of features or samples, whichever is smaller, and at least 1.
    n_comps = int(min(50, X_scaled.shape[1] - 1, X_scaled.shape[0] - 1))
    if n_comps < 1:
        n_comps = 1  # Ensure at least one component is computed

    pca = PCA(n_components=n_comps, random_state=0)
    X_emb = pca.fit_transform(X_scaled)

    # 5. Store the final low-dimensional, batch-corrected embedding.
    adata.obsm["X_emb"] = X_emb

    # 6. Return the AnnData object with the new embedding.
    return adata