import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and log-transform the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure adata.X is a dense numpy array for robust processing.
    # scanpy's pp.normalize_total and pp.log1p typically convert to dense float32,
    # but explicit conversion adds robustness.
    X_processed = adata.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed, dtype=np.float32) # Ensure float32 for consistency

    # Prepare batch labels for subsequent corrections
    batch_labels = adata.obs["batch"].values
    unique_batches = np.unique(batch_labels)

    # 2. Per-gene batch mean subtraction in expression space (before PCA)
    # This step aligns with the guidance "subtract each BATCH's MEAN per gene in EXPRESSION space, BEFORE PCA".
    # It centers the expression of each gene within each batch, removing batch-specific gene expression shifts.
    X_corrected_expr_space = np.copy(X_processed) # Work on a copy of the processed data

    for batch in unique_batches:
        batch_mask = (batch_labels == batch)
        # Calculate the mean expression for each gene within this batch.
        # This results in a vector of shape (n_genes,).
        batch_gene_means = X_corrected_expr_space[batch_mask].mean(axis=0)
        # Subtract these gene means from all cells belonging to this batch.
        X_corrected_expr_space[batch_mask] -= batch_gene_means

    # 3. Perform PCA on the expression data that has been corrected in expression space.
    # Determine n_components safely: at most 20 (as per common practice and problem context),
    # but not more than (n_samples-1) or (n_features-1).
    # PCA requires n_components <= min(n_samples, n_features).
    n_samples, n_features = X_corrected_expr_space.shape
    n_comps = min(20, n_features - 1, n_samples - 1)
    if n_comps < 1: # Ensure at least one component if data is too small or uniform
        n_comps = 1

    pca = PCA(n_components=n_comps, random_state=0)
    initial_emb = pca.fit_transform(X_corrected_expr_space)

    # 4. Batch effect correction in EMBEDDING space (after PCA).
    # This is the "BEST simple method" described in the prompt:
    # subtracting each batch's mean embedding vector (batch-centering in embedding space).
    # This further ensures that batch centroids in the embedding space are aligned.
    corrected_emb = np.copy(initial_emb)

    for batch in unique_batches:
        batch_mask = (batch_labels == batch)
        # Calculate the mean embedding (centroid) for cells in this batch.
        # This results in a vector of shape (n_components,).
        batch_centroid = initial_emb[batch_mask].mean(axis=0)
        # Subtract the batch centroid from all cells in this batch.
        corrected_emb[batch_mask] -= batch_centroid

    # 5. Store the final batch-corrected embedding in adata.obsm["X_emb"].
    adata.obsm["X_emb"] = corrected_emb

    return adata