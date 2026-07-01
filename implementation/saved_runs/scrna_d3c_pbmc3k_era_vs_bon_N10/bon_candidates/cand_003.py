import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    """
    Eliminates batch effects from single-cell RNA-seq data by performing
    normalization, log-transformation, PCA, and then subtracting the
    per-batch mean in the PCA embedding space.

    Args:
        adata: An AnnData object with raw counts in .X and batch labels
               in adata.obs["batch"].
        config: A dictionary for configuration (not used in this specific
                implementation but required by the signature).

    Returns:
        An AnnData object with the batch-corrected low-dimensional embedding
        stored in adata.obsm["X_emb"].
    """
    adata = adata.copy()

    # 1. Preprocessing: Normalize and log-transform the counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure X is a dense numpy array for PCA, handling potential sparse matrices
    X_processed = adata.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed)

    # Determine the number of PCA components safely.
    # We aim for a common practice like 20 components, but limit it by
    # the dimensions of the data to prevent errors with small datasets.
    n_cells, n_genes = X_processed.shape
    n_comps = int(min(20, n_genes - 1, n_cells - 1))
    
    # Ensure n_comps is at least 1, as PCA requires n_components >= 1
    if n_comps < 1:
        n_comps = 1
    
    # 2. Apply PCA to get an initial low-dimensional embedding
    # Using a fixed random_state for reproducibility.
    pca = PCA(n_components=n_comps, random_state=0)
    initial_emb = pca.fit_transform(X_processed)

    # 3. Subtract each batch's mean vector in the embedding space
    # This step centers each batch in the embedding, thereby reducing
    # batch-specific shifts.
    corrected_emb = initial_emb.copy()
    
    for batch_label in adata.obs["batch"].unique():
        # Identify cells belonging to the current batch
        batch_indices = (adata.obs["batch"] == batch_label).to_numpy() # Convert to numpy array for boolean indexing

        # Calculate the mean embedding vector for this batch
        batch_mean_emb = initial_emb[batch_indices].mean(axis=0)

        # Subtract the batch mean from all cells in this batch
        corrected_emb[batch_indices] -= batch_mean_emb

    # Store the final batch-corrected embedding in adata.obsm
    adata.obsm["X_emb"] = corrected_emb

    return adata