import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and Log-transform
    # These steps are standard for scRNA-seq data and are part of the
    # "STRONG, RELIABLE DIRECTION" identified in the problem description.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Prepare data for further processing.
    # PCA from scikit-learn and other numpy operations expect a dense numpy array.
    # adata.X might be sparse (e.g., csr_matrix) after scanpy preprocessing.
    X_processed = adata.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed, dtype=np.float32)

    # 2. Per-gene batch mean subtraction in expression space (before PCA)
    # This step aims to remove batch-specific nuisance shifts from each gene's expression
    # directly in the feature space. This is a recommended "small safe add-on" and
    # preferred over sc.pp.regress_out when explicit numpy implementation is safer.
    batches = adata.obs["batch"].values
    unique_batches = np.unique(batches)

    # Operate on a copy to apply gene-level batch correction
    X_batch_corrected_gene_level = np.copy(X_processed)

    for batch_label in unique_batches:
        batch_mask = (batches == batch_label)
        if np.any(batch_mask):
            # Calculate the mean of each gene for cells in the current batch
            # This results in a vector of shape (n_genes,)
            batch_gene_means = np.mean(X_processed[batch_mask, :], axis=0)
            # Subtract this batch-specific mean from all cells belonging to this batch
            X_batch_corrected_gene_level[batch_mask, :] -= batch_gene_means

    # 3. Scaling (Z-scoring) of genes
    # After batch correction at the gene level, apply standard scaling (mean 0, variance 1)
    # to each gene. This ensures that all genes contribute equally to the PCA
    # and prevents highly variable genes from dominating the principal components.
    # This is a "simple sklearn preprocessing" step.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_batch_corrected_gene_level)

    # 4. Dimensionality reduction using PCA
    n_samples, n_features = X_scaled.shape
    
    # Handle the rare edge case where there's only one sample, PCA is not meaningful.
    if n_samples <= 1:
        # Return a trivial 1-dimensional embedding, fulfilling the shape requirement.
        adata.obsm["X_emb"] = np.zeros((n_samples, 1), dtype=np.float32)
        return adata

    # Calculate n_comps, ensuring it's positive and does not exceed data dimensions.
    # Sticking to 20 components as in the parent, which proved effective for PBMC3k.
    n_comps = min(20, n_samples - 1, n_features) 
    n_comps = max(1, n_comps) # Ensure at least 1 component is always requested.

    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X_scaled) # Compute PCA embedding

    # 5. Batch effect correction in the embedding space (mean subtraction per batch)
    # This step centers the cells of each batch around the origin in the PCA embedding space.
    # This is highlighted in the problem description as the "BEST simple method"
    # for batch mixing while preserving biological variation.
    # This is applied to the embedding `emb` (after PCA).
    emb_corrected = np.copy(emb) # Create a copy of the embedding to modify

    for batch_label in unique_batches:
        batch_mask = (batches == batch_label)
        if np.any(batch_mask):
            # Calculate the mean of embeddings for the current batch
            batch_mean_embedding = np.mean(emb[batch_mask], axis=0)
            # Subtract this batch-specific mean from all embeddings belonging to this batch
            emb_corrected[batch_mask] -= batch_mean_embedding

    # Store the final batch-corrected embedding in adata.obsm
    adata.obsm["X_emb"] = emb_corrected

    return adata