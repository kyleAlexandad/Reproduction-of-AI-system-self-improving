import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    """
    Performs batch integration on single-cell RNA-seq data by combining gene-wise
    batch correction with PCA and subsequent batch-centering in the embedding space.

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
    # This stabilizes variance and makes gene expression distributions more Gaussian-like.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Get the gene expression matrix and ensure it's a dense NumPy array for efficient processing.
    # This handles potential sparse inputs from AnnData.
    X_processed = adata.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed)  # Ensure it's a numpy array

    # 2. Gene-wise Batch Mean Subtraction (New step for improved batch removal)
    # For each gene, we subtract the mean expression within each batch. This removes
    # batch-specific mean differences at the gene level before dimensionality reduction,
    # helping PCA to capture biological variance more cleanly.
    X_corrected_genes = X_processed.copy()
    batch_labels = adata.obs["batch"]

    for batch in batch_labels.unique():
        batch_mask = (batch_labels == batch)
        # Ensure there are cells for the current batch to avoid errors with empty slices.
        if np.sum(batch_mask) > 0:
            # Calculate the mean expression for all genes within this specific batch
            batch_gene_means = X_processed[batch_mask, :].mean(axis=0)
            # Subtract these batch-specific gene means from all cells in this batch
            X_corrected_genes[batch_mask, :] -= batch_gene_means

    # 3. Determine the number of principal components for the embedding.
    # We aim for a moderately low-dimensional space (e.g., 50 components) that captures
    # significant biological variance. Ensure n_comps is at least 1 and less than
    # the number of features or samples, whichever is smaller.
    n_comps = int(min(50, X_corrected_genes.shape[1] - 1, X_corrected_genes.shape[0] - 1))
    if n_comps < 1:
        n_comps = 1  # Ensure at least one component is computed

    # 4. Apply PCA to the gene-wise batch-corrected matrix.
    # This step reduces dimensionality, now operating on data where major gene-level
    # batch effects have already been mitigated.
    pca = PCA(n_components=n_comps, random_state=0)
    X_pca = pca.fit_transform(X_corrected_genes)

    # 5. Batch-centering in the PCA space.
    # This step further refines the batch correction by aligning batch centroids
    # in the low-dimensional embedding, addressing any residual batch effects
    # not fully captured by the gene-wise correction or PCA.
    corrected_embedding = X_pca.copy()
    for batch in batch_labels.unique():
        batch_mask = (batch_labels == batch)
        if np.sum(batch_mask) > 0:
            batch_mean_pca = X_pca[batch_mask, :].mean(axis=0)
            corrected_embedding[batch_mask, :] -= batch_mean_pca

    # 6. Store the final batch-corrected embedding in adata.obsm["X_emb"].
    adata.obsm["X_emb"] = corrected_embedding

    # 7. Return the AnnData object with the new embedding.
    return adata