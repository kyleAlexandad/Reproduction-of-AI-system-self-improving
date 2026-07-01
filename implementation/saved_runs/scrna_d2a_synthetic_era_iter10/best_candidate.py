import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    """
    Performs batch integration on single-cell RNA-seq data using PCA followed by batch-centering.

    Args:
        adata: An AnnData object containing raw gene-expression counts in adata.X and
               batch labels in adata.obs["batch"].
        config: A dictionary for potential configuration parameters (not used in this solution).

    Returns:
        An AnnData object with a new low-dimensional embedding stored in adata.obsm["X_emb"].
        The embedding will have batch effects reduced and biological structure preserved.
    """
    # Create a copy to avoid modifying the original AnnData object in place.
    adata = adata.copy()

    # 1. Preprocessing: Normalize total counts per cell and then log-transform.
    # This stabilizes variance and makes gene expression distributions more Gaussian-like.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 2. Get the gene expression matrix and ensure it's a dense NumPy array.
    # PCA from scikit-learn works best with dense arrays, and this handles potential sparse inputs.
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)  # Ensure it's a numpy array, even if already dense

    # 3. Determine the number of principal components for the initial embedding.
    # We aim for a moderately low-dimensional space (e.g., 50 components) that captures
    # significant biological variance. Ensure n_comps is at least 1 and less than
    # the number of features or samples, whichever is smaller.
    n_comps = int(min(50, X.shape[1] - 1, X.shape[0] - 1))
    if n_comps < 1:
        n_comps = 1  # Ensure at least one component is computed

    # 4. Apply PCA to get an initial low-dimensional embedding.
    # This step reduces dimensionality and captures the main sources of variance,
    # including both biological signal and batch effects.
    pca = PCA(n_components=n_comps, random_state=0)
    X_pca = pca.fit_transform(X)

    # 5. Batch-centering in the PCA space.
    # For each batch, calculate the mean of its cells in the PCA embedding and subtract
    # this mean from all cells belonging to that batch. This effectively shifts
    # batch centroids to the origin, thereby mixing batches in the embedding space.
    batch_labels = adata.obs["batch"]
    corrected_embedding = X_pca.copy()  # Initialize with the PCA embedding

    for batch in batch_labels.unique():
        batch_mask = (batch_labels == batch)
        # Ensure there are cells for the current batch to avoid errors with empty slices.
        if np.sum(batch_mask) > 0:
            batch_mean = X_pca[batch_mask, :].mean(axis=0)
            corrected_embedding[batch_mask, :] -= batch_mean

    # 6. Store the batch-corrected embedding in adata.obsm["X_emb"].
    adata.obsm["X_emb"] = corrected_embedding

    # 7. Return the AnnData object with the new embedding.
    return adata