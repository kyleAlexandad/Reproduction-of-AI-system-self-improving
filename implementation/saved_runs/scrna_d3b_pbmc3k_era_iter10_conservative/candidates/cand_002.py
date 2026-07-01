import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and Log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Convert to dense array and ensure float32 type for numerical stability,
    # direct numpy array manipulation, and sklearn compatibility.
    X_processed = adata.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed, dtype=np.float32)

    # 2. Batch effect correction in EXPRESSION space (per-gene mean subtraction per batch)
    # This step aims to remove batch-specific shifts in gene expression means before PCA.
    # It centers each gene's expression distribution independently for each batch.
    batches = adata.obs["batch"].values
    unique_batches = np.unique(batches)

    for batch_label in unique_batches:
        batch_mask = (batches == batch_label)
        if np.any(batch_mask):
            # Calculate the mean expression for each gene within the current batch
            batch_gene_means = np.mean(X_processed[batch_mask, :], axis=0)
            # Subtract these means from the expression values of cells in this batch
            X_processed[batch_mask, :] -= batch_gene_means

    # 3. Dimensionality reduction using PCA
    # Determine n_components dynamically.
    # Ensure n_components is at least 1 and less than min(n_samples, n_features)
    # to avoid errors with very small datasets or few features.
    n_comps = int(min(20, X_processed.shape[1] - 1, X_processed.shape[0] - 1))
    n_comps = max(1, n_comps) # Ensure at least 1 component

    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X_processed)

    # 4. Batch effect correction in the EMBEDDING space (mean subtraction per batch)
    # This step further aligns batches by centering their means in the reduced
    # dimensional space, compensating for any residual batch differences.
    emb_corrected = np.copy(emb) # Work on a copy of the embedding

    for batch_label in unique_batches:
        batch_mask = (batches == batch_label)
        if np.any(batch_mask):
            # Calculate the mean of embeddings for the current batch
            batch_mean_embedding = np.mean(emb[batch_mask], axis=0)
            # Subtract the batch mean from all embeddings belonging to this batch
            emb_corrected[batch_mask] -= batch_mean_embedding

    # Store the final batch-corrected embedding
    adata.obsm["X_emb"] = emb_corrected

    return adata