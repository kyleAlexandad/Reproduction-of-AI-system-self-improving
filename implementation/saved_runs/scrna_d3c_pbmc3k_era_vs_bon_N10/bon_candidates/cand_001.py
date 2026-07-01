import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize total counts and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 2. Prepare data for PCA: Ensure data is a dense numpy array
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    # Determine number of PCA components.
    # It must be at least 1 and less than min(n_cells, n_genes).
    # The value 20 is a common heuristic for scRNA-seq embeddings.
    n_comps = int(min(20, X.shape[1] - 1, X.shape[0] - 1))
    
    # Ensure n_comps is at least 1 to avoid errors with PCA
    if n_comps < 1:
        n_comps = 1

    # 3. Perform PCA to obtain an initial low-dimensional embedding
    emb_pca = PCA(n_components=n_comps, random_state=0).fit_transform(X)

    # 4. Batch centering in embedding space: Subtract each batch's mean vector
    # This is the core batch effect correction step.
    emb_corrected = emb_pca.copy()  # Initialize corrected embedding with PCA output
    
    batches = adata.obs["batch"].values
    unique_batches = np.unique(batches)

    for batch_label in unique_batches:
        # Identify cells belonging to the current batch
        batch_mask = (batches == batch_label)
        
        # Calculate the mean of the PCA embedding for cells in this batch
        batch_mean_embedding = emb_pca[batch_mask].mean(axis=0)
        
        # Subtract the batch mean from all cells in this batch
        emb_corrected[batch_mask] -= batch_mean_embedding

    # 5. Store the final batch-corrected embedding
    adata.obsm["X_emb"] = emb_corrected

    return adata