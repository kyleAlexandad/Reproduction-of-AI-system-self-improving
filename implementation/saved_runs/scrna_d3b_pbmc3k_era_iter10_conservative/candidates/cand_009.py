import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and Log-transform
    # These steps are standard for scRNA-seq data.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Prepare data for further processing.
    # Convert adata.X to a dense numpy array if it's sparse, as required by
    # subsequent numpy operations and sklearn PCA.
    X_processed = adata.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed, dtype=np.float32)

    n_samples, n_features = X_processed.shape

    # Handle the edge case where there's only one sample. PCA is not meaningful.
    if n_samples <= 1:
        # Return a trivial 1-dimensional embedding, fulfilling the shape requirement.
        adata.obsm["X_emb"] = np.zeros((n_samples, 1), dtype=np.float32)
        return adata

    # 2. Batch effect correction in expression space (per-gene, per-batch mean subtraction)
    # This step is added to the parent candidate based on the "simple, reliable building blocks"
    # guidance, which suggests "subtract each BATCH's MEAN per gene in EXPRESSION space, BEFORE PCA".
    # This helps to remove gene-level batch-specific shifts before dimensionality reduction.
    batches = adata.obs["batch"].values
    unique_batches = np.unique(batches)

    X_corrected_gene_level = np.copy(X_processed) # Work on a copy of the processed data

    for batch in unique_batches:
        batch_mask = (batches == batch)
        if np.any(batch_mask):
            # Calculate the mean expression for each gene within the current batch.
            # The result is a vector of shape (n_genes,).
            batch_mean_expression = np.mean(X_processed[batch_mask, :], axis=0)
            # Subtract this batch-specific mean from all cells in this batch, across all genes.
            X_corrected_gene_level[batch_mask, :] -= batch_mean_expression

    # 3. Dimensionality reduction using PCA
    # Determine the number of components for PCA.
    # We use a practical limit (e.g., 20 components) common in single-cell analysis.
    # Ensure n_components is valid: at least 1, and does not exceed min(n_samples-1, n_features).
    n_comps = min(20, n_samples - 1, n_features)
    n_comps = max(1, n_comps) # Ensure at least 1 component is always requested.

    pca = PCA(n_components=n_comps, random_state=0)
    # Perform PCA on the gene-level batch-corrected data.
    emb = pca.fit_transform(X_corrected_gene_level)

    # 4. Batch effect correction in the embedding space (mean subtraction per batch)
    # This step centers the cells of each batch around the origin in the PCA embedding space.
    # This is highlighted in the problem description as the "BEST simple method"
    # for batch mixing while preserving biological variation.
    emb_corrected = np.copy(emb) # Create a copy of the embedding to modify

    for batch in unique_batches:
        batch_mask = (batches == batch)
        if np.any(batch_mask):
            # Calculate the mean of embeddings for the current batch.
            # The result is a vector of shape (n_comps,).
            batch_mean_embedding = np.mean(emb[batch_mask], axis=0)
            # Subtract this batch-specific mean from all embeddings belonging to this batch.
            emb_corrected[batch_mask] -= batch_mean_embedding

    # Store the final batch-corrected embedding in adata.obsm
    adata.obsm["X_emb"] = emb_corrected

    return adata