import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and Log-transform expression data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure data is dense for PCA and converted to float32
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # 2. Apply PCA to obtain an initial low-dimensional embedding
    # The number of components is set robustly, similar to the parent candidate,
    # and a common choice for single-cell data (20-50 PCs).
    n_comps = int(min(20, X.shape[1] - 1, X.shape[0] - 1))
    
    pca = PCA(n_components=n_comps, random_state=0)
    emb_initial = pca.fit_transform(X)

    # 3. Batch Correction: Subtract per-batch mean in the embedding space
    # This step centers each batch's embedding distribution around the origin,
    # thereby reducing batch-specific shifts in the embedding.
    batches = adata.obs["batch"].unique()
    emb_corrected = emb_initial.copy()

    for batch in batches:
        # Identify cells belonging to the current batch
        batch_mask = (adata.obs["batch"] == batch)
        
        # Extract the initial embeddings for cells in this batch
        batch_embedding_subset = emb_initial[batch_mask]
        
        # Calculate the mean embedding vector for this specific batch
        batch_mean = np.mean(batch_embedding_subset, axis=0)
        
        # Subtract the batch mean from all cells within this batch
        emb_corrected[batch_mask] -= batch_mean

    # Store the batch-corrected embedding
    adata.obsm["X_emb"] = emb_corrected

    return adata