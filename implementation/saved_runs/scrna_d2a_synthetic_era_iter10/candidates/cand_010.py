import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def eliminate_batch_effect_fn(adata, config):
    """
    Performs batch integration on single-cell RNA-seq data using gene-level batch mean
    correction, global gene scaling, and PCA. This approach aims to remove batch
    effects at the feature (gene) level before dimensionality reduction, ensuring
    PCA primarily captures biological variation while preserving overall gene expression levels.

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
    # This is a standard step to stabilize variance and make gene expression
    # distributions more amenable to downstream linear methods.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 2. Get the gene expression matrix and ensure it's a dense NumPy array.
    # scikit-learn transformers work most efficiently with dense arrays.
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32) # Use float32 for memory efficiency and speed

    # 3. Gene-level Batch Mean Correction while preserving overall gene means.
    # For each gene, and for each batch, we subtract the batch-specific mean
    # and then re-add the global mean of that gene. This removes batch-specific
    # shifts in gene expression means while maintaining the global expression
    # profile of each gene.
    batch_labels = adata.obs["batch"]
    X_corrected_batch_means = X.copy()

    # Calculate the overall mean of each gene across all cells before any batch correction.
    global_gene_means = X.mean(axis=0)

    # Iterate through each unique batch.
    for batch in batch_labels.unique():
        batch_mask = (batch_labels == batch)
        if np.sum(batch_mask) > 0: # Ensure there are cells for the current batch
            # Calculate the mean expression for each gene within the current batch.
            batch_gene_means = X[batch_mask, :].mean(axis=0)
            
            # Subtract the batch-specific gene mean from cells in this batch.
            # This centers the gene expression within each batch.
            X_corrected_batch_means[batch_mask, :] -= batch_gene_means
            
            # Re-add the original global gene means to cells in this batch.
            # This ensures that while batch-to-batch differences for each gene are removed,
            # the average expression level of a gene across *all* cells is preserved.
            X_corrected_batch_means[batch_mask, :] += global_gene_means

    # 4. Global Gene Scaling (Z-scoring).
    # Standardize gene expression across all cells (after batch correction) to have
    # a mean of 0 and a standard deviation of 1. This is crucial for PCA, as it
    # ensures that all genes contribute equally to the variance calculation,
    # preventing highly variable genes from dominating the principal components.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_corrected_batch_means)

    # 5. Determine the number of principal components.
    # We aim for a moderately low-dimensional space (e.g., 50 components) that
    # captures significant biological variance. Ensure n_comps is at least 1
    # and less than the number of features or samples, whichever is smaller.
    n_comps = int(min(50, X_scaled.shape[1] - 1, X_scaled.shape[0] - 1))
    if n_comps < 1:
        n_comps = 1

    # 6. Apply PCA to get the final low-dimensional embedding.
    # PCA will now operate on the data where batch effects at the gene-level
    # have been largely removed and genes are standardized. This should yield
    # components that primarily reflect biological variation.
    pca = PCA(n_components=n_comps, random_state=0)
    X_emb = pca.fit_transform(X_scaled)

    # 7. Store the batch-corrected and dimensionally reduced embedding.
    adata.obsm["X_emb"] = X_emb

    # 8. Return the AnnData object with the new embedding.
    return adata