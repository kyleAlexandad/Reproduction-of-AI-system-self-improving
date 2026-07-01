import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def eliminate_batch_effect_fn(adata, config):
    """
    Performs batch integration on single-cell RNA-seq data by applying a two-stage process:
    1. Standard preprocessing (normalization, log-transformation) followed by initial PCA
       to reduce dimensionality and denoise the data.
    2. Linear regression-based batch effect removal directly on the principal components.
       For each component, the linear effect of batch is modeled and subtracted,
       producing a batch-corrected low-dimensional embedding.

    Args:
        adata: An AnnData object containing raw gene-expression counts in adata.X and
               batch labels in adata.obs["batch"].
        config: A dictionary for potential configuration parameters (not used in this solution).

    Returns:
        An AnnData object with a new low-dimensional embedding stored in adata.obsm["X_emb"].
        The embedding will have batch effects reduced and biological structure preserved.
    """
    # Create a copy to avoid modifying the original AnnData object in place.
    adata = adata.copy()

    # 1. Preprocessing: Normalize total counts per cell and then log-transform.
    # This standard step helps to stabilize variance and make gene expression distributions
    # more amenable to linear models and dimensionality reduction.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure adata.X is a dense NumPy array. This is important for scikit-learn
    # compatibility and performance with subsequent operations, as `sc.pp.pca`
    # can sometimes produce sparse intermediates if not handled.
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    
    # 2. Initial Dimensionality Reduction using PCA.
    # We apply PCA to get an initial low-dimensional representation. This step helps
    # to denoise the data and focus on major axes of variation before explicit batch correction.
    # The number of components is chosen to be moderate (e.g., 50),
    # but also capped by the actual dimensions of the data to prevent errors.
    initial_n_comps = int(min(50, adata.X.shape[1] - 1, adata.X.shape[0] - 1))
    if initial_n_comps < 1:
        initial_n_comps = 1  # Ensure at least one component is computed

    # `sc.pp.pca` computes PCA and stores the result in `adata.obsm["X_pca"]`.
    # `random_state` ensures reproducibility.
    sc.pp.pca(adata, n_comps=initial_n_comps, random_state=0)
    X_pca = adata.obsm["X_pca"]

    # 3. Batch Correction in the PCA space using linear regression.
    # For each principal component, we model and remove the linear effect of batch.
    # This approach is more general than simple mean subtraction and helps to
    # align the embedding across different batches more effectively.
    
    batch_labels = adata.obs["batch"]
    
    # Create a design matrix for the batch effect using one-hot encoding.
    # `drop_first=False` creates a dummy variable for every batch. `LinearRegression`
    # with `fit_intercept=True` can handle the potential multicollinearity gracefully
    # by effectively determining the unique contributions.
    batch_design_matrix = pd.get_dummies(batch_labels, prefix='batch').astype(float)

    # Initialize an array to store the batch-corrected embedding.
    corrected_embedding = np.empty_like(X_pca)
    
    # Iterate through each principal component (column in X_pca) and apply linear regression.
    for i in range(X_pca.shape[1]):
        y_component = X_pca[:, i]  # Get the values for the current principal component

        # Fit a linear model: `y_component ~ batch_design_matrix`.
        # The model estimates how much each batch contributes to the variation in this specific PC.
        # `n_jobs=-1` uses all available CPU cores for fitting, speeding up the process.
        model = LinearRegression(fit_intercept=True, n_jobs=-1)
        model.fit(batch_design_matrix, y_component)
        
        # Predict the batch-related variation for this component based on the fitted model.
        predicted_batch_effect = model.predict(batch_design_matrix)
        
        # Subtract the predicted batch effect from the original component values.
        # The resulting residuals represent the component with batch effects removed,
        # thereby preserving the biological variation that is not explained by batch.
        corrected_embedding[:, i] = y_component - predicted_batch_effect

    # 4. Store the final batch-corrected low-dimensional embedding in adata.obsm["X_emb"].
    # This embedding aims to mix batches while preserving biological signals.
    adata.obsm["X_emb"] = corrected_embedding

    # 5. Return the AnnData object with the new embedding.
    return adata