import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and Log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Convert to dense array for processing and sklearn PCA if it's sparse
    # Ensure float32 for consistency and potential memory savings
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # 2. Gene-level batch effect correction (mean subtraction per gene within each batch)
    # This step implements: "subtract each BATCH's MEAN per gene in EXPRESSION space, BEFORE PCA"
    batches = adata.obs["batch"].values
    unique_batches = np.unique(batches)

    # Create a copy of the expression matrix to modify
    X_corrected_gene_level = np.copy(X)

    for batch in unique_batches:
        batch_mask = (batches == batch)
        if np.any(batch_mask):
            # Calculate the mean expression for each gene for cells in the current batch
            batch_gene_means = np.mean(X[batch_mask, :], axis=0)
            # Subtract this mean from the expression of cells belonging to this batch
            X_corrected_gene_level[batch_mask, :] -= batch_gene_means
            
    # The matrix `X_corrected_gene_level` will now be used for PCA
    X_for_pca = X_corrected_gene_level

    # 3. Dimensionality reduction using PCA
    # Determine n_components, ensuring it's valid.
    # Must be at least 1, and less than min(n_samples, n_features) of the input data.
    n_comps = int(min(20, X_for_pca.shape[1] - 1, X_for_pca.shape[0] - 1))
    n_comps = max(1, n_comps) # Ensure at least 1 component

    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X_for_pca)

    # 4. Embedding-level batch effect correction (mean subtraction per batch in embedding space)
    # This step implements: "subtract each BATCH's MEAN vector in EMBEDDING space, AFTER PCA"
    # This is often the most effective simple method for batch correction.
    emb_corrected = np.copy(emb)

    for batch in unique_batches:
        batch_mask = (batches == batch)
        
        if np.any(batch_mask):
            # Calculate the mean of embeddings for the current batch
            batch_mean_embedding = np.mean(emb[batch_mask], axis=0)
            
            # Subtract the batch mean embedding from all embeddings belonging to this batch
            emb_corrected[batch_mask] -= batch_mean_embedding

    # Store the final batch-corrected embedding in adata.obsm
    adata.obsm["X_emb"] = emb_corrected

    return adata