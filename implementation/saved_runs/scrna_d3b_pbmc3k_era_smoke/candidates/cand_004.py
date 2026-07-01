import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    """
    Integrates single-cell RNA-seq data by removing artificial batch effects
    while preserving biological structure, using ComBat on highly variable genes
    followed by PCA.

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
    # This stabilizes the variance and makes gene expression distributions more Gaussian-like.
    sc.pp.normalize_total(adata_integrated, target_sum=1e4)
    sc.pp.log1p(adata_integrated)

    # 2. Feature selection: Identify highly variable genes (HVGs)
    # Focusing on HVGs helps to reduce noise and computational load,
    # and these genes are often most relevant for biological signal.
    sc.pp.highly_variable_genes(adata_integrated, n_top_genes=2000)
    
    # Create a subset AnnData object containing only the highly variable genes
    adata_hvg = adata_integrated[:, adata_integrated.var.highly_variable].copy()

    # Ensure the data matrix is dense for ComBat, as it often performs better
    # with dense matrices and may convert internally anyway.
    if hasattr(adata_hvg.X, 'toarray'):
        adata_hvg.X = adata_hvg.X.toarray()

    # 3. Batch Effect Removal: Apply ComBat to the highly variable genes
    # ComBat is an empirical Bayes method specifically designed to adjust for
    # batch effects in gene expression data, directly addressing per-gene nuisance shifts.
    # It modifies adata_hvg.X in place with the batch-corrected expression values.
    sc.tl.combat(adata_hvg, key='batch')

    # 4. Scale the data: Z-score each batch-corrected gene
    # This step ensures that genes with higher expression values do not
    # disproportionately influence the subsequent dimensionality reduction.
    # Clipping extreme values (max_value=10) improves robustness to outliers.
    sc.pp.scale(adata_hvg, max_value=10)

    # 5. Dimensionality Reduction: Principal Component Analysis (PCA)
    # PCA reduces the data to a low-dimensional embedding that captures the
    # most variance in the batch-corrected gene expression, effectively
    # summarizing the biological signal.
    n_comps = min(50, adata_hvg.X.shape[1] - 1, adata_hvg.X.shape[0] - 1)
    n_comps = max(1, n_comps) # Ensure at least one component

    # Use sklearn's PCA for explicit control and reproducibility
    pca = PCA(n_components=n_comps, svd_solver='arpack', random_state=0)
    embedding = pca.fit_transform(adata_hvg.X)

    # Store the resulting low-dimensional batch-integrated embedding
    # in adata.obsm["X_emb"] as required.
    adata.obsm["X_emb"] = embedding

    return adata