import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    """
    Performs batch integration on single-cell RNA-seq data using a robust preprocessing
    pipeline including highly variable gene selection, ComBat batch correction,
    scaling, and PCA.

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
    # This stabilizes variance and makes gene expression distributions more Gaussian-like,
    # which is beneficial for downstream linear methods like ComBat and PCA.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 2. Highly Variable Gene (HVG) selection.
    # Selecting HVGs helps to focus on biologically relevant variance and filter out noise.
    # Using 'batch_key' in HVG selection helps to identify HVGs that are truly variable
    # independent of batch effects, preventing batch-specific noise from being amplified.
    # 'flavor="seurat_v3"' is a robust method for HVG selection.
    sc.pp.highly_variable_genes(
        adata,
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
        n_top_genes=2000,
        batch_key="batch",  # Account for batch differences during HVG selection
        flavor="seurat_v3"
    )

    # Create a temporary AnnData object containing only the highly variable genes.
    # This reduces computational load for subsequent steps and focuses on key biological signals.
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    # Ensure the expression matrix is dense. While scanpy functions often handle sparse
    # matrices, explicit conversion can sometimes optimize performance for `combat`
    # and `PCA` with certain `scikit-learn` versions.
    if hasattr(adata_hvg.X, "toarray"):
        adata_hvg.X = adata_hvg.X.toarray()
    else:
        adata_hvg.X = np.asarray(adata_hvg.X) # Ensure it's a NumPy array

    # 3. Batch Effect Correction using ComBat.
    # ComBat (sc.pp.combat) is a well-established method for removing linear batch effects
    # by adjusting for additive and multiplicative batch effects across genes.
    # It operates directly on the expression matrix (adata_hvg.X).
    sc.pp.combat(adata_hvg, key="batch")

    # 4. Scale the data.
    # Z-score-like scaling (mean 0, variance 1) per gene ensures that highly expressed
    # genes do not dominate the PCA and all genes contribute equally regardless of their
    # magnitude. 'max_value' clips extreme values, which can improve robustness against outliers.
    sc.pp.scale(adata_hvg, max_value=10)

    # 5. Dimensionality Reduction using PCA.
    # PCA is applied to the batch-corrected and scaled highly variable genes.
    # This step captures the principal sources of remaining variance, which should now
    # primarily represent biological signal after batch effect removal.
    X_to_pca = adata_hvg.X

    # Determine the number of principal components.
    # A common choice is 50, but we ensure it's not more than the number of features
    # (highly variable genes) or samples, whichever is smaller, and is at least 1.
    n_comps = int(min(50, X_to_pca.shape[1] - 1, X_to_pca.shape[0] - 1))
    if n_comps < 1:
        n_comps = 1  # Ensure at least one component is computed

    pca = PCA(n_components=n_comps, random_state=0)
    X_emb = pca.fit_transform(X_to_pca)

    # 6. Store the resulting embedding in adata.obsm["X_emb"].
    # This embedding is the final batch-corrected, low-dimensional representation.
    # It is attached to the original (copied) AnnData object, ensuring it contains
    # all cells from the input.
    adata.obsm["X_emb"] = X_emb

    # 7. Return the AnnData object with the new embedding.
    return adata