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
    adata_processed = adata.copy() # Operate on a copy to avoid modifying the original adata prematurely

    # 1. Preprocessing: Normalize total counts per cell and log-transform
    # This is a standard first step for scRNA-seq data to account for differences in
    # sequencing depth per cell and to stabilize variance, making gene expression
    # more comparable across cells.
    sc.pp.normalize_total(adata_processed, target_sum=1e4)
    sc.pp.log1p(adata_processed)

    # 2. Feature selection: Identify highly variable genes (HVGs)
    # Focusing on highly variable genes helps to filter out technical noise and
    # retain genes that are most likely to contribute to biological differences.
    # 'n_top_genes=2000' is a common and effective choice for many datasets.
    # The 'seurat' flavor is generally robust.
    sc.pp.highly_variable_genes(adata_processed, n_top_genes=2000, flavor='seurat')
    
    # Create a subset AnnData object containing only the highly variable genes.
    # This reduces computational cost and focuses downstream analyses on informative features.
    adata_hvg = adata_processed[:, adata_processed.var.highly_variable].copy()

    # 3. Batch Effect Removal: ComBat
    # ComBat is a robust and widely used method for correcting batch effects.
    # It models and adjusts for both additive and multiplicative batch effects (mean and variance shifts)
    # across batches for each gene. It modifies `adata_hvg.X` in place.
    # The 'batch' column from `adata_processed.obs` is automatically carried over to `adata_hvg.obs`.
    sc.tl.combat(adata_hvg, key='batch')

    # 4. Scale the data: Z-score each gene
    # Scaling ensures that genes with higher overall expression or variance do not
    # disproportionately influence subsequent dimensionality reduction steps.
    # Clipping extreme values (max_value=10) helps to improve robustness to outliers.
    sc.pp.scale(adata_hvg, max_value=10)

    # 5. Dimensionality Reduction: Principal Component Analysis (PCA)
    # PCA is applied to reduce the high-dimensional gene expression data into a
    # lower-dimensional embedding, capturing the most significant sources of variation
    # after batch correction and scaling.
    # Determine the number of components, capped at 50, but also respecting data dimensions.
    n_comps = min(50, adata_hvg.X.shape[1] - 1, adata_hvg.X.shape[0] - 1)
    n_comps = max(1, n_comps) # Ensure at least one component for the embedding

    pca = PCA(n_components=n_comps, svd_solver='arpack', random_state=0)
    # Compute the initial low-dimensional embedding from the batch-corrected and scaled data.
    initial_embedding = pca.fit_transform(adata_hvg.X)

    # 6. Post-PCA Batch Centering: A refinement step to further improve batch mixing
    # This technique directly addresses any residual batch effect in the low-dimensional space.
    # It works by calculating the mean embedding vector for each batch and then
    # subtracting this mean from all cells belonging to that batch. This centers
    # each batch's data cloud in the embedding space, encouraging better mixing
    # of batch cells while preserving the internal biological structure within batches.
    final_embedding = initial_embedding.copy()
    
    # Retrieve unique batch labels from the observation data (from the full adata_processed object).
    batches = adata_processed.obs["batch"].unique()

    for batch_label in batches:
        # Create a boolean mask to identify cells belonging to the current batch.
        batch_mask = (adata_processed.obs["batch"] == batch_label).values
        
        # Calculate the mean of the embeddings for all cells in this specific batch.
        batch_mean_embedding = initial_embedding[batch_mask].mean(axis=0)
        
        # Subtract the calculated batch mean from the embeddings of all cells in this batch.
        final_embedding[batch_mask] -= batch_mean_embedding

    # Store the final, batch-integrated low-dimensional embedding in the original AnnData object.
    adata.obsm["X_emb"] = final_embedding

    return adata