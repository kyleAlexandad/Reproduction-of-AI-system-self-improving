import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
import scanpy.external.pp as pepp

def eliminate_batch_effect_fn(adata, config):
    adata_integrated = adata.copy()

    # 1. Preprocessing: Normalize total counts per cell and log-transform
    sc.pp.normalize_total(adata_integrated, target_sum=1e4)
    sc.pp.log1p(adata_integrated)

    # 2. Feature selection: Identify highly variable genes
    # Using 'seurat_v3' flavor which is generally robust for log-normalized data.
    # n_top_genes=2000 is a common choice for datasets of this size to capture biological signal.
    sc.pp.highly_variable_genes(adata_integrated, n_top_genes=2000, flavor='seurat_v3')
    
    # Subset the AnnData object to only include highly variable genes
    adata_hvg = adata_integrated[:, adata_integrated.var.highly_variable].copy()

    # 3. Batch Effect Removal: Combat
    # Combat is an empirical Bayes framework for batch effect correction.
    # It is often more powerful than simple linear regression (like scanpy's regress_out)
    # as it models both mean and variance batch effects for each gene, borrowing
    # information across genes, which can lead to better batch mixing and biology preservation.
    # It operates on log-transformed data and directly modifies adata_hvg.X.
    pepp.combat(adata_hvg, key='batch', inplace=True)

    # 4. Scale the data: Z-score each gene to prevent highly expressed genes
    # from dominating the PCA. Clipping values (max_value=10) can improve
    # robustness to outliers caused by the batch correction or original data.
    # This step is applied after Combat to standardize gene contributions to PCA.
    sc.pp.scale(adata_hvg, max_value=10)

    # 5. Dimensionality Reduction: Principal Component Analysis (PCA)
    # Determine the number of components for PCA.
    # Cap at 50, but ensure it's not greater than the number of features or samples minus one,
    # and at least 1. This makes it robust to very small datasets or few HVGs.
    n_comps = min(50, adata_hvg.X.shape[1] - 1, adata_hvg.X.shape[0] - 1)
    n_comps = max(1, n_comps) # Ensure at least one component

    # Perform PCA. The 'arpack' solver is efficient for a fixed number of components.
    pca = PCA(n_components=n_comps, svd_solver='arpack', random_state=0)
    embedding = pca.fit_transform(adata_hvg.X)

    # Store the resulting low-dimensional embedding in adata.obsm["X_emb"]
    # The result must be stored in the original `adata` object.
    adata.obsm["X_emb"] = embedding

    return adata