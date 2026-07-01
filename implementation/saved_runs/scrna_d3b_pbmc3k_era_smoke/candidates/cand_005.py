import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    """
    Integrates single-cell RNA-seq data by removing artificial batch effects
    while preserving biological structure. This implementation combines
    per-gene linear regression for batch effect removal with post-PCA
    batch-centering in the embedding space, aiming for improved batch mixing
    and biological signal preservation.

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
    # Create a copy to avoid modifying the original AnnData object until the end
    adata_integrated = adata.copy()

    # 1. Preprocessing: Normalize total counts per cell and log-transform
    # This step ensures comparability across cells and stabilizes variance.
    sc.pp.normalize_total(adata_integrated, target_sum=1e4)
    sc.pp.log1p(adata_integrated)

    # 2. Feature selection: Identify highly variable genes
    # Focusing on highly variable genes helps to concentrate on biological signal
    # and reduces computational load. Using 'seurat_v3' flavor for robustness.
    sc.pp.highly_variable_genes(adata_integrated, n_top_genes=2000, flavor='seurat_v3')
    
    # Subset the AnnData object to only include highly variable genes for subsequent steps
    adata_hvg = adata_integrated[:, adata_integrated.var.highly_variable].copy()

    # 3. Batch Effect Removal (per-gene): Regress out the batch effect
    # This step performs a linear regression for each gene against the batch
    # indicator variables and stores the residuals. This removes linear
    # batch-specific shifts for individual genes.
    # Using n_jobs=-1 for potential parallelization on available CPU cores.
    sc.pp.regress_out(adata_hvg, 'batch', n_jobs=-1)

    # 4. Scale the data: Z-score each gene
    # Scaling to unit variance ensures that highly expressed genes do not
    # disproportionately influence the PCA. Clipping helps against outliers.
    sc.pp.scale(adata_hvg, max_value=10)

    # 5. Dimensionality Reduction: Principal Component Analysis (PCA)
    # Perform PCA on the batch-corrected and scaled gene expression matrix
    # to obtain an initial low-dimensional embedding.
    n_comps = min(50, adata_hvg.X.shape[1] - 1, adata_hvg.X.shape[0] - 1)
    n_comps = max(1, n_comps) # Ensure at least one component for PCA

    pca = PCA(n_components=n_comps, svd_solver='arpack', random_state=0)
    embedding_initial = pca.fit_transform(adata_hvg.X)

    # 6. Batch Effect Removal (in embedding space): Subtract batch means
    # This step further refines batch integration by aligning the centroids
    # of each batch in the low-dimensional embedding space. This helps to
    # mix batches more thoroughly after the initial per-gene correction.
    embedding_corrected = embedding_initial.copy()
    
    # Get unique batch labels from the original adata object (cell order is preserved)
    batches = adata.obs["batch"].unique()
    
    for batch_label in batches:
        # Identify cells belonging to the current batch
        batch_indices = np.where(adata.obs["batch"] == batch_label)[0]
        
        if len(batch_indices) > 0:
            # Calculate the mean of the embedding vectors for this batch
            batch_mean = embedding_initial[batch_indices].mean(axis=0)
            
            # Subtract the calculated batch mean from all cells in this batch
            # This effectively centers each batch's embedding around the overall mean.
            embedding_corrected[batch_indices] -= batch_mean

    # Store the final batch-integrated low-dimensional embedding
    adata.obsm["X_emb"] = embedding_corrected

    return adata