import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Determine a safe fallback value for NaNs and empty contexts.
    # Prioritize the last *finite* value in context. If no finite values, use 0.0.
    safe_nan_replacement = 0.0
    if n > 0:
        finite_context_values = context[np.isfinite(context)]
        if len(finite_context_values) > 0:
            # Use the mean of finite context values as a robust fallback.
            safe_nan_replacement = np.mean(finite_context_values)
        # else: safe_nan_replacement remains 0.0

    if n == 0:
        return np.full(prediction_length, safe_nan_replacement, dtype=float)

    season = 24  # Daily seasonality for hourly data (period 24)
    
    # Fallback for very short contexts: repeat the last available finite value.
    if n < season:
        val_to_repeat = float(context[-1]) if np.isfinite(context[-1]) else safe_nan_replacement
        return np.full(prediction_length, val_to_repeat, dtype=float)

    # Calculate seasonal component by averaging last K full seasons.
    # K is the number of full seasons to average, capped at 8 for stability and performance.
    # It must be at least 1 if n >= season.
    K = min(max(1, n // season), 8)
    
    # Extract the last K * season points and reshape into K rows, `season` columns.
    # Each row represents a full season.
    # Use np.nanmean to handle any NaNs within the historical seasonal data gracefully.
    mat = context[-K * season:].reshape(K, season)
    seasonal_pattern = np.nanmean(mat, axis=0)
    
    # Ensure seasonal_pattern contains no NaNs (e.g., if a whole column in `mat` was NaN).
    seasonal_pattern = np.nan_to_num(seasonal_pattern, nan=safe_nan_replacement)

    # Calculate a damped level adjustment based on the most recent observation's deviation
    # from its expected seasonal value.
    last_idx_in_season = (n - 1) % season
    current_seasonal_expected_value = seasonal_pattern[last_idx_in_season]
    
    # If context[-1] is NaN, use its seasonal expectation to avoid NaN in level_correction.
    last_observed_value = float(context[-1]) if np.isfinite(context[-1]) else current_seasonal_expected_value
    
    # The deviation of the last observed point from its seasonal expectation.
    level_correction = last_observed_value - current_seasonal_expected_value
    
    # Apply a damping factor to the level correction to make it more conservative.
    damped_level_correction = level_correction * 0.5 
    
    # Tile the seasonal pattern to cover the prediction length.
    reps = int(np.ceil(prediction_length / season))
    seasonal_forecast = np.tile(seasonal_pattern, reps)[:prediction_length]

    # Add the damped level correction to the seasonal forecast.
    out = seasonal_forecast + damped_level_correction

    # Final check: ensure no NaN or inf values are returned. Replace with the robust safe_nan_replacement.
    return np.nan_to_num(out, nan=safe_nan_replacement).astype(float)