import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and Log-transform gene expression data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Convert adata.X to a dense NumPy array and ensure float type for direct operations.
    # This handles potential sparse matrices from previous steps.
    X_processed = adata.X
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed, dtype=np.float32)

    # Extract batch labels for subsequent corrections
    batches = adata.obs["batch"].values
    unique_batches = np.unique(batches)

    # --- 2. Subtract each BATCH's MEAN per gene in EXPRESSION space, BEFORE PCA ---
    # This step aims to remove additive batch effects on a gene-by-gene basis,
    # making gene expression distributions more comparable across batches before PCA.
    # It centers each batch's gene mean towards the global gene mean.

    # Calculate the global mean expression for each gene across all cells.
    global_gene_means = np.mean(X_processed, axis=0)

    # Create a copy of the processed expression data to apply gene-wise batch correction.
    X_corrected_gene_batch = np.copy(X_processed)

    for batch_label in unique_batches:
        batch_mask = (batches == batch_label)
        if np.any(batch_mask):  # Ensure there are cells for the current batch
            # Calculate the mean expression for each gene specifically within this batch.
            batch_gene_means = np.mean(X_processed[batch_mask, :], axis=0)

            # Determine the batch effect for each gene: the difference between its
            # batch-specific mean and its global mean.
            # Subtracting this effectively adjusts the batch's gene means to align
            # with the global gene means.
            batch_effect_per_gene = batch_gene_means - global_gene_means

            # Apply this correction to all cells belonging to the current batch.
            X_corrected_gene_batch[batch_mask, :] -= batch_effect_per_gene

    # --- 3. Dimensionality reduction using PCA ---
    # Apply PCA to the gene-wise batch-corrected expression data to obtain a
    # lower-dimensional embedding.
    n_samples, n_features = X_corrected_gene_batch.shape

    # Determine the number of PCA components. A common heuristic is 20, but we
    # must ensure it's valid (at least 1, and less than the number of features/samples).
    n_comps = int(min(20, n_features - 1, n_samples - 1))
    n_comps = max(1, n_comps)  # Ensure at least 1 component to avoid PCA errors.

    pca = PCA(n_components=n_comps, random_state=0)
    # The result 'emb' is the initial low-dimensional embedding.
    emb = pca.fit_transform(X_corrected_gene_batch)

    # --- 4. Batch effect correction in the EMBEDDING space (mean subtraction per batch) ---
    # This is a critical step, identified as the "BEST simple method" in the guidance.
    # It centers the embeddings of cells from each batch, promoting better mixing in the
    # low-dimensional space.

    # Create a copy of the embedding to store the batch-corrected version.
    emb_corrected = np.copy(emb)

    for batch_label in unique_batches:
        batch_mask = (batches == batch_label)  # Re-use the batch masks.
        if np.any(batch_mask):  # Ensure there are cells for the current batch
            # Calculate the mean vector of embeddings for all cells in the current batch.
            batch_mean_embedding = np.mean(emb[batch_mask, :], axis=0)

            # Subtract this batch-specific mean from the embeddings of all cells
            # belonging to this batch.
            emb_corrected[batch_mask, :] -= batch_mean_embedding

    # Store the final, batch-corrected low-dimensional embedding in adata.obsm.
    adata.obsm["X_emb"] = emb_corrected

    return adata