import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    """
    Integrates single-cell RNA-seq data by removing artificial batch effects
    while preserving biological structure.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with:
        - adata.X               : raw gene-expression counts (n_cells, n_genes).
        - adata.obs["batch"]    : categorical batch label.
        - adata.obs["cell_type"]: HIDDEN, not to be used.
    config : dict
        Configuration dictionary (not used in this implementation but included for signature compliance).

    Returns
    -------
    adata : AnnData
        The input AnnData object with a new key `adata.obsm["X_emb"]` containing
        the low-dimensional batch-integrated embedding.
    """
    adata_integrated = adata.copy()

    # 1. Preprocessing: Normalize total counts per cell and log-transform
    sc.pp.normalize_total(adata_integrated, target_sum=1e4)
    sc.pp.log1p(adata_integrated)

    # 2. Feature selection: Identify highly variable genes
    # This step focuses on genes most likely to carry biological signal
    # and reduces the dimensionality for subsequent steps.
    sc.pp.highly_variable_genes(adata_integrated, n_top_genes=2000)
    
    # Subset the AnnData object to only include highly variable genes
    adata_hvg = adata_integrated[:, adata_integrated.var.highly_variable].copy()

    # 3. Batch Effect Removal: Regress out the batch effect
    # This step performs a linear regression for each gene to remove the
    # contribution of the 'batch' variable from its expression.
    # It directly modifies adata_hvg.X to contain the residuals.
    sc.pp.regress_out(adata_hvg, 'batch')

    # 4. Scale the data: Z-score each gene to prevent highly expressed genes
    # from dominating the PCA. Clipping values (max_value=10) can improve
    # robustness to outliers.
    sc.pp.scale(adata_hvg, max_value=10)

    # 5. Dimensionality Reduction: Principal Component Analysis (PCA)
    # Determine the number of components for PCA.
    # We cap it at 50, but ensure it's not greater than the number of features
    # or samples minus one, and at least 1.
    n_comps = min(50, adata_hvg.X.shape[1] - 1, adata_hvg.X.shape[0] - 1)
    n_comps = max(1, n_comps) # Ensure at least one component

    # Perform PCA. The 'arpack' solver is often efficient for fixed n_components.
    # We use sklearn's PCA directly for explicit control and compatibility,
    # as scanpy's tl.pca is also a wrapper around sklearn.
    pca = PCA(n_components=n_comps, svd_solver='arpack', random_state=0)
    embedding = pca.fit_transform(adata_hvg.X)

    # Store the resulting low-dimensional embedding in adata.obsm["X_emb"]
    adata.obsm["X_emb"] = embedding

    return adata