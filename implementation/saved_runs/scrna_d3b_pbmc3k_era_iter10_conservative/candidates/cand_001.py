import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and Log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Convert to dense array for sklearn PCA if it's sparse
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32) # Ensure float type

    # 2. Dimensionality reduction using PCA
    # Determine n_components, ensuring it's valid (at least 1, and less than min(n_samples, n_features))
    n_comps = int(min(20, X.shape[1] - 1, X.shape[0] - 1))
    n_comps = max(1, n_comps) # Ensure at least 1 component

    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X)

    # 3. Batch effect correction in the embedding space (mean subtraction per batch)
    batches = adata.obs["batch"].values
    unique_batches = np.unique(batches)

    # Create a copy of the embedding to modify
    emb_corrected = np.copy(emb)

    for batch in unique_batches:
        # Find cells belonging to the current batch
        batch_mask = (batches == batch)
        
        # Calculate the mean of embeddings for the current batch
        # Only proceed if there are cells in this batch
        if np.any(batch_mask):
            batch_mean = np.mean(emb[batch_mask], axis=0)
            
            # Subtract the batch mean from all embeddings belonging to this batch
            emb_corrected[batch_mask] -= batch_mean

    # Store the batch-corrected embedding
    adata.obsm["X_emb"] = emb_corrected

    return adata