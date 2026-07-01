import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    # Create a copy to work on, avoiding modification of the original AnnData object
    adata_integrated = adata.copy()

    # Preprocessing: Normalize total counts per cell and then log-transform
    sc.pp.normalize_total(adata_integrated, target_sum=1e4)
    sc.pp.log1p(adata_integrated)

    # Ensure the expression matrix is a dense numpy array for PCA
    X = adata_integrated.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32) # Ensure float32 for consistency and memory if needed

    # Determine the number of PCA components.
    # Use a maximum of 20 components, but respect dataset dimensions.
    # Ensure at least 1 component if possible.
    n_cells, n_genes = X.shape
    n_comps = min(20, n_genes - 1, n_cells - 1)
    if n_comps < 1:
        n_comps = 1 # Fallback to 1 component if data is too small

    # Perform PCA to get an initial low-dimensional embedding
    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X)

    # Batch effect removal: Subtract per-batch mean in the embedding space
    # This step centers each batch's embedding distribution, mixing them while preserving
    # within-batch (biological) variance.
    batches = adata_integrated.obs["batch"].unique()
    for batch_id in batches:
        # Identify cells belonging to the current batch
        batch_mask = (adata_integrated.obs["batch"] == batch_id).values
        
        # Calculate the mean embedding vector for this batch
        batch_mean_emb = emb[batch_mask].mean(axis=0)
        
        # Subtract the batch mean from the embeddings of all cells in this batch
        emb[batch_mask] -= batch_mean_emb

    # Store the integrated, batch-corrected embedding in .obsm["X_emb"]
    adata_integrated.obsm["X_emb"] = emb
    
    return adata_integrated