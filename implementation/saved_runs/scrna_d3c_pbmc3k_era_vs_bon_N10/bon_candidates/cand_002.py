import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    # Create a copy to avoid modifying the original AnnData object in place for intermediate steps
    adata = adata.copy()

    # 1. Preprocessing: Normalize total counts and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure adata.X is a dense numpy array for PCA.
    # It might be sparse initially.
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32) # Ensure data is float32 for consistency

    # Determine the number of components for PCA.
    # It's good practice to limit this based on dataset size and avoid too many components.
    # Use min of 20, or (number of genes - 1), or (number of cells - 1) to be safe.
    n_comps = int(min(20, X.shape[1] - 1, X.shape[0] - 1))
    
    # Ensure n_comps is at least 1 to satisfy the requirement 1 <= d.
    if n_comps < 1:
        n_comps = 1

    # 2. Perform PCA to get an initial low-dimensional embedding
    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X)

    # 3. Batch effect correction in embedding space (centroid alignment)
    # This step subtracts the mean embedding vector of each batch from all cells in that batch,
    # effectively centering each batch's centroid at the origin in the embedding space.
    batches = adata.obs["batch"].values
    unique_batches = np.unique(batches)

    for batch_id in unique_batches:
        # Create a boolean mask for cells belonging to the current batch
        batch_mask = (batches == batch_id)
        
        # Calculate the mean of the embeddings for cells in this batch
        batch_mean_embedding = emb[batch_mask].mean(axis=0)
        
        # Subtract this mean from all cells belonging to this batch
        emb[batch_mask] -= batch_mean_embedding

    # 4. Store the batch-corrected embedding in adata.obsm
    adata.obsm["X_emb"] = emb

    return adata