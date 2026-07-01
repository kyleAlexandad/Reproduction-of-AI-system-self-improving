import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    # Create a copy to avoid modifying the original AnnData object in place
    adata_processed = adata.copy()

    # --- Preprocessing steps as recommended ---
    # 1. Normalize total counts per cell to a target sum
    sc.pp.normalize_total(adata_processed, target_sum=1e4)
    # 2. Log-transform the data
    sc.pp.log1p(adata_processed)

    # Get the processed data matrix, ensuring it's a dense numpy array
    X = adata_processed.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)  # Ensure float32 for consistency and potential memory savings

    # --- Dimensionality Reduction (PCA) ---
    # Determine the number of PCA components.
    # Use a maximum of 20 components, but ensure it's not more than the number of features or samples - 1
    # This prevents errors on very small datasets.
    n_comps = int(min(20, X.shape[1] - 1, X.shape[0] - 1))
    if n_comps < 1:  # Ensure at least 1 component if possible
        n_comps = 1

    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X)

    # --- Batch Effect Correction in Embedding Space ---
    # This is the core improvement: subtract each batch's mean vector in the embedding space
    batch_labels = adata_processed.obs["batch"]
    unique_batches = batch_labels.unique()

    # Create a copy of the embedding to store the corrected version
    corrected_emb = np.copy(emb)

    for batch in unique_batches:
        # Identify cells belonging to the current batch
        batch_mask = (batch_labels == batch)

        # Calculate the mean embedding for this batch
        batch_mean_embedding = np.mean(emb[batch_mask], axis=0)

        # Subtract the batch mean from all cells within this batch
        corrected_emb[batch_mask] -= batch_mean_embedding

    # --- Store the corrected embedding ---
    adata_processed.obsm["X_emb"] = corrected_emb

    return adata_processed