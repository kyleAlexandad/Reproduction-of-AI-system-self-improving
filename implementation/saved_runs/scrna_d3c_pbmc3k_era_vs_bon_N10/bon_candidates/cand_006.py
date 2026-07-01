import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Normalize total counts per cell to 1e4 and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Convert the expression matrix to a dense numpy array if it's sparse.
    # This is necessary for sklearn.decomposition.PCA.
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)  # Ensure it's a numpy array

    # 2. Perform Principal Component Analysis (PCA)
    # Determine the number of components robustly.
    # PCA's n_components must be 0 < n_components <= min(n_samples, n_features).
    # The problem requires the embedding dimension d to be 1 <= d <= n_genes.
    # We cap at 20 components as recommended for lightweight methods.
    # max(1, ...) ensures at least one component, fulfilling the d >= 1 requirement.
    n_comps = max(1, min(20, X.shape[0], X.shape[1]))

    # Use a random_state for reproducibility, potentially from the config.
    pca = PCA(n_components=n_comps, random_state=config.get("seed", 0))
    emb = pca.fit_transform(X)

    # 3. Batch-centering in the PCA embedding space
    # This step subtracts the mean of each batch from its corresponding cells
    # in the low-dimensional embedding, effectively mixing batches.
    batch_labels = adata.obs["batch"]
    emb_corrected = emb.copy()

    for batch_id in batch_labels.unique():
        # Identify cells belonging to the current batch
        batch_mask = (batch_labels == batch_id)

        # Calculate the mean vector for this batch in the embedding space
        batch_mean = emb[batch_mask].mean(axis=0)

        # Subtract the batch mean from all cells of this batch
        emb_corrected[batch_mask] -= batch_mean

    # Store the corrected embedding in adata.obsm["X_emb"]
    adata.obsm["X_emb"] = emb_corrected

    return adata