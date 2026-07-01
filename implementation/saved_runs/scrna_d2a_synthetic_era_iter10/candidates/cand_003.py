import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    """
    Performs batch integration on single-cell RNA-seq data by preprocessing,
    selecting highly variable genes, regressing out batch effects, scaling,
    and then applying PCA to obtain a low-dimensional embedding.

    Args:
        adata: An AnnData object containing raw gene-expression counts in adata.X and
               batch labels in adata.obs["batch"].
        config: A dictionary for potential configuration parameters (not used in this solution).

    Returns:
        An AnnData object with a new low-dimensional embedding stored in adata.obsm["X_emb"].
        The embedding aims to have reduced batch effects and preserved biological structure.
    """
    # Create a copy to avoid modifying the original AnnData object in place.
    adata = adata.copy()

    # 1. Preprocessing: Normalize total counts per cell and then log-transform.
    # This stabilizes variance and makes gene expression distributions more suitable for linear models.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 2. Highly Variable Gene (HVG) selection.
    # Focuses the analysis on genes with significant biological variation, reducing noise
    # and computational load. 'seurat_v3' flavor is generally robust.
    # The `batch_key` argument for `highly_variable_genes` can be used to identify HVGs
    # within each batch, which can be useful for integration, but for simplicity and
    # focusing on `regress_out` for batch correction, we'll keep it standard here.
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')

    # Subset the AnnData object to include only the highly variable genes.
    adata = adata[:, adata.var.highly_variable]

    # 3. Regress out the batch effect.
    # This step directly removes the linear influence of the 'batch' covariate from the
    # gene expression matrix (adata.X). It's a powerful method for batch correction
    # at the gene level before dimensionality reduction.
    sc.pp.regress_out(adata, 'batch')

    # 4. Scale data to unit variance and center to zero, clipping values.
    # This is crucial for PCA, as it ensures that each gene contributes equally
    # and prevents highly expressed genes or outliers from dominating the components.
    # `max_value=10` helps to clip extreme values, increasing robustness.
    sc.pp.scale(adata, max_value=10)

    # 5. Get the processed gene expression matrix.
    # Ensure it's a dense NumPy array, as scikit-learn's PCA works best with dense data.
    X_processed = adata.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed)  # Ensure it's a numpy array

    # 6. Apply PCA to get the final low-dimensional embedding.
    # The number of components is set to a reasonable default (100) but limited by
    # the actual number of features (genes) or samples to avoid errors.
    # Since `regress_out` and `scale` have already handled much of the variance,
    # a moderate number of components should be sufficient to capture biological signal.
    n_comps = int(min(100, X_processed.shape[1] - 1, X_processed.shape[0] - 1))
    if n_comps < 1:
        n_comps = 1  # Ensure at least one component is computed

    pca = PCA(n_components=n_comps, random_state=0)
    X_emb = pca.fit_transform(X_processed)

    # 7. Store the batch-corrected, low-dimensional embedding in adata.obsm["X_emb"].
    adata.obsm["X_emb"] = X_emb

    # 8. Return the AnnData object with the new embedding.
    return adata