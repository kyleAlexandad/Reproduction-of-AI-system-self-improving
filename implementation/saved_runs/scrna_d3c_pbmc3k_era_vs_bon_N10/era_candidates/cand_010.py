import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()

    # 1. Preprocessing: Normalize and log-transform the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure adata.X is a dense numpy array for robust processing.
    # This step is important as sc.pp operations can sometimes keep sparse arrays,
    # and direct numpy operations/PCA expect dense arrays.
    X_initial_processed = adata.X
    if hasattr(X_initial_processed, "toarray"):
        X_initial_processed = X_initial_processed.toarray()
    X_initial_processed = np.asarray(X_initial_processed, dtype=np.float32)

    batch_labels = adata.obs["batch"].values
    unique_batches = np.unique(batch_labels)

    # 2. Batch effect correction in EXPRESSION space (before PCA):
    # This step directly addresses the suggestion to "subtract each BATCH's MEAN per gene
    # in EXPRESSION space, BEFORE PCA". This helps remove global batch-specific shifts
    # in gene expression prior to dimensionality reduction.
    X_expression_batch_corrected = np.copy(X_initial_processed)
    for batch in unique_batches:
        batch_mask = (batch_labels == batch)
        # Calculate the mean expression for each gene within this batch
        batch_gene_means = X_initial_processed[batch_mask].mean(axis=0)
        # Subtract these batch-specific means from all cells in this batch
        X_expression_batch_corrected[batch_mask] -= batch_gene_means

    # 3. Robust scaling / z-scoring of genes:
    # This step addresses the suggestion for "robust scaling / z-scoring of genes
    # (sc.pp.scale(max_value=10) or sklearn StandardScaler)".
    # It standardizes gene expression to have mean 0 and variance 1, and clips
    # extreme values to prevent outliers from dominating PCA.
    # We update adata.X with the expression-level batch corrected data before scaling.
    adata.X = X_expression_batch_corrected
    sc.pp.scale(adata, max_value=10)

    # Ensure adata.X is dense and float32 after scaling, ready for PCA.
    X_for_pca = adata.X
    if hasattr(X_for_pca, "toarray"):
        X_for_pca = X_for_pca.toarray()
    X_for_pca = np.asarray(X_for_pca, dtype=np.float32)

    # 4. Perform PCA to get an initial low-dimensional embedding.
    # Determine n_components safely: at most 20 (as per common practice),
    # but not more than (n_samples-1) or (n_features-1).
    # PCA requires n_components <= min(n_samples, n_features).
    n_samples, n_features = X_for_pca.shape
    n_comps = min(20, n_features - 1, n_samples - 1)
    if n_comps < 1:
        n_comps = 1 # Ensure at least one component if data is too small

    pca = PCA(n_components=n_comps, random_state=0)
    initial_emb = pca.fit_transform(X_for_pca)

    # 5. Batch effect correction in EMBEDDING space (after PCA):
    # This is the "BEST simple method" identified in the prompt,
    # subtracting each batch's mean embedding vector (batch-centering in embedding space).
    # This step provides a final refinement for batch mixing within the lower-dimensional embedding.
    corrected_emb = np.copy(initial_emb)
    # batch_labels are already available from the initial processing.
    for batch in unique_batches:
        batch_mask = (batch_labels == batch)
        # Calculate the mean embedding (centroid) for cells in this batch.
        batch_centroid = initial_emb[batch_mask].mean(axis=0)
        # Subtract the batch centroid from all cells in this batch.
        corrected_emb[batch_mask] -= batch_centroid

    # 6. Store the final batch-corrected embedding in adata.obsm["X_emb"].
    adata.obsm["X_emb"] = corrected_emb

    return adata