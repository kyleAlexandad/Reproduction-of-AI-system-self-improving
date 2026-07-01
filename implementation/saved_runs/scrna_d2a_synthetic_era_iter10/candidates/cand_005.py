import numpy as np
import scanpy as sc


def eliminate_batch_effect_fn(adata, config):
    """
    Performs batch integration on single-cell RNA-seq data.

    This function implements a robust batch integration strategy:
    1. Standard preprocessing: normalization and log-transformation.
    2. Highly variable gene (HVG) selection to focus on biologically relevant genes.
    3. Linear regression to explicitly remove batch effects for each gene.
    4. Scaling the corrected gene expression to ensure equal contribution in PCA.
    5. Principal Component Analysis (PCA) for dimensionality reduction, resulting in
       a low-dimensional embedding with reduced batch effects and preserved biological signal.

    Args:
        adata: An AnnData object containing raw gene-expression counts in adata.X and
               categorical batch labels in adata.obs["batch"].
        config: A dictionary for potential configuration parameters, e.g.,
                'n_comps' for the number of PCA components (default: 50).

    Returns:
        An AnnData object with a new low-dimensional embedding stored in adata.obsm["X_emb"].
        The embedding aims to have batch effects reduced and biological structure preserved.
    """
    # Create a copy to avoid modifying the original AnnData object in place.
    adata = adata.copy()

    # 1. Preprocessing: Normalize total counts per cell and then log-transform.
    # This step stabilizes variance and makes gene expression distributions more suitable
    # for downstream linear models and dimensionality reduction.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 2. Identify Highly Variable Genes (HVGs).
    # Focusing on HVGs helps to filter out noise and emphasize genes that contribute
    # most to biological variability. 'subset=True' modifies adata in place to
    # keep only the selected HVGs.
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, subset=True)

    # 3. Regress out batch effects from gene expression.
    # This crucial step uses linear regression for each gene to model and subtract
    # the variance explained by the 'batch' covariate. It's a direct way to handle
    # additive batch effects, aiming to leave behind the biological signal.
    # Scanpy's regress_out adds the global mean back after regression.
    sc.pp.regress_out(adata, 'batch')

    # 4. Scale the data.
    # Scaling ensures that genes (after batch effect removal) with naturally higher
    # expression variance do not disproportionately influence the subsequent PCA.
    # 'max_value' clips extreme values, mitigating the impact of outliers.
    sc.pp.scale(adata, max_value=10)

    # 5. Determine the number of principal components for the embedding.
    # We aim for a moderately low-dimensional space (e.g., 50 components).
    # The number of genes (adata.X.shape[1]) might have changed due to HVG selection.
    # The number of components must be less than min(n_samples, n_features).
    current_n_genes = adata.X.shape[1]
    current_n_cells = adata.X.shape[0]

    # Handle edge case where no genes or cells remain after filtering/preprocessing.
    if current_n_genes == 0 or current_n_cells == 0:
        # Return an empty embedding if no valid data to compute PCA on.
        adata.obsm["X_emb"] = np.empty((current_n_cells, 0), dtype=np.float32)
        return adata

    target_n_comps = config.get('n_comps', 50)
    # Ensure n_comps is positive and does not exceed data dimensions.
    n_comps = int(min(target_n_comps, current_n_genes - 1, current_n_cells - 1))
    if n_comps < 1:
        # If possible, always compute at least one component.
        n_comps = 1

    # 6. Apply PCA to get the final low-dimensional embedding.
    # This step reduces dimensionality while capturing the main sources of remaining
    # variance, which should now primarily represent biological differences.
    sc.tl.pca(adata, n_comps=n_comps, random_state=0)

    # 7. Store the batch-corrected and dimensionally reduced embedding in adata.obsm["X_emb"].
    # sc.tl.pca stores its results in adata.obsm["X_pca"].
    adata.obsm["X_emb"] = adata.obsm["X_pca"]

    # 8. Return the AnnData object with the new embedding.
    return adata