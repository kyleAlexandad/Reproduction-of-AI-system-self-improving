import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Determine a robust fallback value. This is used for short contexts,
    # and to replace any potential NaNs/Infs in the final forecast.
    # It ensures that even if context[-1] is NaN, we have a sensible numeric fallback.
    robust_fallback_value = np.nan_to_num(context[-1], nan=0.0, posinf=0.0, neginf=0.0)
    
    # Define the dominant seasonality.
    # For hourly data, daily seasonality (24 hours) is critical as per benchmark facts.
    season = 24

    # Handle short context: if not enough data for a full season,
    # fall back to a naive forecast using the robust last observed value.
    if n < season:
        return np.full(prediction_length, robust_fallback_value, dtype=float)

    # Determine K, the number of past full seasons to average.
    # Averaging multiple past seasons helps to denoise the seasonal pattern.
    # Cap K at 8 to avoid using excessively old data and keep computation lightweight.
    # Ensure K is at least 1 if there's enough data for at least one season.
    K = min(max(1, n // season), 8)
    
    # Extract the last K * season points from the context.
    # This slice captures the most recent full seasonal cycles.
    # Reshape it into a K x season matrix, where each row represents a full season.
    mat = context[-K * season:].reshape(K, season)
    
    # Compute the average seasonal pattern across the K seasons.
    # Use np.nanmean to safely handle potential NaNs within the historical `mat`.
    # `seasonal` will be a 1D array of length `season`, representing the typical value for each hour.
    seasonal = np.nanmean(mat, axis=0)

    # If the seasonal pattern itself could not be computed (e.g., all `K * season` values were NaN),
    # fallback to filling the prediction with the robust fallback value.
    if np.all(np.isnan(seasonal)):
        return np.full(prediction_length, robust_fallback_value, dtype=float)

    # Project the seasonal pattern over the `prediction_length`.
    # `reps` determines how many times the seasonal pattern needs to be tiled to cover the horizon.
    reps = int(np.ceil(prediction_length / season))
    out = np.tile(seasonal, reps)[:prediction_length]
    
    # Ensure no NaNs or Infs in the final output.
    # Any NaN, positive infinity, or negative infinity in the forecast is replaced
    # by our `robust_fallback_value`.
    out = np.nan_to_num(out, nan=robust_fallback_value, posinf=robust_fallback_value, neginf=robust_fallback_value)
    
    return out.astype(float)