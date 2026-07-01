import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def eliminate_batch_effect_fn(adata, config):
    # Create a copy to avoid modifying the input AnnData object directly
    adata = adata.copy()

    # 1. Preprocessing: Normalize total counts per cell and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Get the expression matrix and ensure it's a dense NumPy array for linear algebra operations
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32) # Ensure float32 for consistency and memory efficiency

    # Handle edge case where there's no data or too few features/observations
    if adata.n_obs == 0 or adata.n_vars == 0:
        # If no valid data, return an empty (n_cells, 1) or (0, 1) array.
        # This satisfies the requirement for a 2D float array.
        adata.obsm["X_emb"] = np.empty((adata.n_obs, 1), dtype=np.float32)
        return adata

    # 2. Per-gene linear batch effect removal
    # This step models and removes the linear effect of batches on each gene's expression,
    # aiming to reduce gene-specific batch effects before dimensionality reduction.
    batch_labels = adata.obs["batch"]
    if batch_labels.dtype.name != 'category':
        batch_labels = batch_labels.astype('category')
    
    # Create a design matrix for batch effects using one-hot encoding.
    # `drop_first=True` prevents multicollinearity with the intercept term in LinearRegression.
    # The intercept then represents the mean of the dropped (first) category.
    design_matrix_batches = pd.get_dummies(batch_labels, drop_first=True, dtype=int)
    design_matrix_batches_np = design_matrix_batches.to_numpy()

    # Only perform batch correction if there is more than one unique batch
    if design_matrix_batches_np.shape[1] > 0: # Checks if there are actual batch indicator variables
        X_corrected = np.zeros_like(X, dtype=np.float32)
        
        # Iterate over each gene (column) to perform linear regression
        for i in range(X.shape[1]):
            gene_expression = X[:, i]
            
            # Fit a linear model: gene_expression = intercept + batch_effects + error
            # fit_intercept=True allows the model to estimate a baseline expression level
            model = LinearRegression(fit_intercept=True, n_jobs=-1) # n_jobs=-1 utilizes all available CPU cores
            model.fit(design_matrix_batches_np, gene_expression)
            
            # Predict the portion of gene expression explained by batch effects (and intercept)
            predicted_effect = model.predict(design_matrix_batches_np)
            
            # Remove the batch-specific deviations while preserving the overall gene mean.
            # This is achieved by subtracting the full prediction and then adding back the intercept.
            # Effectively, it removes the difference in expression attributed to batch
            # relative to the baseline batch (represented by the intercept).
            X_corrected[:, i] = gene_expression - predicted_effect + model.intercept_
        
        # Replace the original expression matrix with the batch-corrected one
        X = X_corrected
    # If there's only one batch, design_matrix_batches_np will be empty, and X remains unchanged,
    # which is the correct behavior as no batch effect correction is needed.

    # 3. Dimensionality Reduction (PCA) on the (batch-corrected) gene expression matrix
    # Determine the number of components for PCA.
    # `max_desired_comps` is an upper bound; actual components are capped by data dimensions.
    # `adata.n_obs - 1` is a robust upper bound for `n_components` to avoid singular matrices.
    max_desired_comps = 30
    n_comps = min(max_desired_comps, adata.n_obs - 1, adata.n_vars)

    # Ensure n_comps is positive for PCA to return a meaningful embedding
    if n_comps <= 0:
        adata.obsm["X_emb"] = np.empty((adata.n_obs, 1), dtype=np.float32)
        return adata
    
    pca = PCA(n_components=n_comps, random_state=0)
    emb = pca.fit_transform(X)

    # 4. Batch Effect Correction in PCA space (batch-centering)
    # This step complements the per-gene correction by aligning the centroids of different
    # batches in the low-dimensional embedding space, further mitigating residual batch shifts.
    if batch_labels.nunique() > 1: # Only apply if there's more than one batch
        global_mean_emb = np.mean(emb, axis=0)

        # Iterate through each unique batch and apply the mean correction
        for batch_id in batch_labels.cat.categories:
            batch_indices = (batch_labels == batch_id).to_numpy() # Boolean mask for cells in current batch
            
            if np.any(batch_indices): # Ensure there are cells belonging to the current batch
                batch_mean_emb = np.mean(emb[batch_indices], axis=0)
                # Subtract batch mean and add global mean to align batch centroids to the overall dataset centroid
                emb[batch_indices] = emb[batch_indices] - batch_mean_emb + global_mean_emb

    # 5. Store the final corrected low-dimensional embedding
    adata.obsm["X_emb"] = emb

    return adata