import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and Log-transform
    # These steps are standard for scRNA-seq data and are part of the
    # "STRONG, RELIABLE DIRECTION" identified in the problem description.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Prepare data for further processing.
    # PCA from scikit-learn and custom numpy operations expect a dense numpy array.
    # adata.X might be sparse (e.g., csr_matrix) after scanpy preprocessing.
    X_processed = adata.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed, dtype=np.float32)

    # 2. Per-gene batch mean subtraction in EXPRESSION space (BEFORE PCA)
    # This step aims to remove a portion of the batch effect directly from the gene expression
    # data, as suggested in the prompt for "small safe add-ons".
    batches = adata.obs["batch"].values
    unique_batches = np.unique(batches)

    # Create a copy of the processed expression data to apply this first batch correction
    X_corrected_expression = np.copy(X_processed)

    for batch_label in unique_batches:
        batch_mask = (batches == batch_label)
        if np.any(batch_mask):
            # Calculate the mean expression for each gene specifically within the current batch
            batch_gene_means = np.mean(X_processed[batch_mask, :], axis=0)
            # Subtract these batch-specific gene means from all cells belonging to this batch.
            # This centers the expression of each gene within each batch around zero.
            X_corrected_expression[batch_mask, :] -= batch_gene_means

    # 3. Dimensionality reduction using PCA
    n_samples, n_features = X_corrected_expression.shape
    
    # Handle the rare edge case where there's only one sample, PCA is not meaningful.
    if n_samples <= 1:
        # Return a trivial 1-dimensional embedding, fulfilling the shape requirement.
        adata.obsm["X_emb"] = np.zeros((n_samples, 1), dtype=np.float32)
        return adata

    # Determine the number of components for PCA.
    # We use a practical limit (e.g., 20 components) common in single-cell analysis.
    # Ensure n_components is valid: at least 1, and does not exceed min(n_samples - 1, n_features).
    # PCA can compute at most min(n_samples-1, n_features) components.
    n_comps = min(20, n_samples - 1, n_features) 
    n_comps = max(1, n_comps) # Ensure at least 1 component is always requested.

    pca = PCA(n_components=n_comps, random_state=0)
    # Perform PCA on the expression data after the initial batch correction
    emb = pca.fit_transform(X_corrected_expression)

    # 4. Batch effect correction in the EMBEDDING space (mean subtraction per batch)
    # This step centers the cells of each batch around the origin in the PCA embedding space.
    # This is highlighted in the problem description as the "BEST simple method"
    # for batch mixing while preserving biological variation.
    emb_corrected = np.copy(emb) # Create a copy of the embedding to modify

    for batch_label in unique_batches: # Re-use unique_batches and batches from step 2
        batch_mask = (batches == batch_label)
        if np.any(batch_mask):
            # Calculate the mean of embeddings for the current batch
            batch_mean_embedding = np.mean(emb[batch_mask], axis=0)
            # Subtract this batch-specific mean from all embeddings belonging to this batch
            emb_corrected[batch_mask] -= batch_mean_embedding

    # Store the final two-stage batch-corrected embedding in adata.obsm
    adata.obsm["X_emb"] = emb_corrected

    return adata