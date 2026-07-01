import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    """
    Performs batch integration on single-cell RNA-seq data by first performing PCA
    and then subtracting batch-specific means in the PCA embedding space.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with:
        - adata.X: Raw gene-expression counts, shape (n_cells, n_genes).
        - adata.obs["batch"]: Categorical batch label.
    config : dict
        A dictionary for configuration parameters (not used in this specific
        implementation but required by the signature).

    Returns
    -------
    AnnData
        The input AnnData object with a new key `adata.obsm["X_emb"]`
        containing the batch-corrected low-dimensional embedding.
        The embedding is a 2D float array of shape (n_cells, d),
        where 1 <= d <= n_genes.
    """
    # 1. Create a copy of adata to avoid modifying the original in place.
    adata = adata.copy()

    # 2. Normalization: Normalize total counts per cell to a target sum (e.g., 1e4).
    # This step is crucial for making counts comparable across cells.
    sc.pp.normalize_total(adata, target_sum=1e4)

    # 3. Log-transform: Apply log1p transformation to the data.
    # This helps in stabilizing variance and making data more Gaussian-like.
    sc.pp.log1p(adata)

    # Prepare data for PCA. Ensure it's a dense NumPy array.
    # AnnData.X can be sparse, so convert to dense if necessary for scikit-learn PCA.
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    # Handle edge case where data might be empty or too small for PCA.
    if adata.n_obs == 0 or adata.n_vars == 0:
        adata.obsm["X_emb"] = np.empty((adata.n_obs, 0))
        return adata

    # Determine the number of PCA components.
    # A common choice is 50, but it should not exceed the number of features or samples - 1.
    # Ensure at least 1 component if data allows.
    n_comps = max(1, min(50, X.shape[1] - 1, X.shape[0] - 1))

    # 4. PCA: Compute an initial low-dimensional embedding.
    # PCA is a fundamental dimensionality reduction technique.
    # Using random_state for reproducibility.
    pca = PCA(n_components=n_comps, random_state=0)
    emb_pca = pca.fit_transform(X)

    # 5. Batch effect removal in embedding space: Subtract each batch's mean embedding.
    # This is the core batch integration step as per the recommended strategy.
    emb_corrected = emb_pca.copy()  # Create a copy to store the corrected embedding

    # Get batch labels from adata.obs.
    batches = adata.obs["batch"]
    unique_batches = batches.unique()

    for batch_label in unique_batches:
        # Identify cells belonging to the current batch
        batch_indices = (batches == batch_label).values

        # Calculate the mean embedding vector for this specific batch
        mean_embedding_batch = emb_pca[batch_indices].mean(axis=0)

        # Subtract the batch-specific mean from all cells in that batch.
        # This centers each batch in the embedding space, reducing batch-specific shifts.
        emb_corrected[batch_indices] -= mean_embedding_batch

    # 6. Store the batch-corrected embedding in adata.obsm["X_emb"].
    adata.obsm["X_emb"] = emb_corrected

    return adata