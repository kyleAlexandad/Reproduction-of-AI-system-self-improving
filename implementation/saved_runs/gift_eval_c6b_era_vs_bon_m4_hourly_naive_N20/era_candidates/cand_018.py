import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- 1. Robust Fallback Value ---
    # This value is crucial for handling empty, very short, or entirely NaN/Inf contexts.
    # It attempts to use the last known finite value from the context, otherwise defaults to 0.0.
    fallback_value = 0.0
    if n > 0:
        # Find indices of all finite values in the context.
        last_finite_idx = np.where(np.isfinite(context))[0]
        if len(last_finite_idx) > 0:
            # If finite values exist, the last one is the most reliable recent value.
            fallback_value = context[last_finite_idx[-1]]
    # Ensure fallback_value itself is finite, handling cases where context might contain only NaNs/Infs.
    if not np.isfinite(fallback_value):
        fallback_value = 0.0

    # Define the primary seasonal period for hourly data.
    # The critical benchmark facts strongly indicate daily seasonality (24 hours) as dominant.
    season_daily = 24

    # --- 2. Handle very short contexts ---
    # If the context is empty, return an array of zeros to satisfy the output length requirement.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # If the context is shorter than a full daily season, we cannot establish a reliable seasonal pattern.
    # In these cases, return a constant forecast using the robust fallback value.
    # This effectively acts as a last-value naive forecast (or 0.0 if no valid last value).
    if n < season_daily:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 3. Core Forecasting Logic: Robust Seasonal Averaging (Period 24) ---
    # This strategy improves upon simple seasonal naive by averaging the last K full daily cycles.
    # This helps to denoise the seasonal pattern, making it more stable and potentially more accurate.

    # Determine K, the number of past full daily seasons to average.
    # K is at least 1 (if n >= season_daily), at most 8, and limited by the available context length.
    K = min(max(1, n // season_daily), 8)
    
    # Extract the relevant portion of the context for averaging.
    # This slice is guaranteed to be valid and contain K full seasons due to the checks above.
    # context[-K * season_daily:] is safe because K * season_daily <= (n // season_daily) * season_daily <= n.
    context_for_seasonal = context[-K * season_daily:]
    
    # Reshape the data into K rows (each representing a day) and `season_daily` columns (each representing an hour).
    mat = context_for_seasonal.reshape(K, season_daily)
    
    # Clean any NaN/Inf values within the matrix before averaging.
    # This prevents these values from propagating into the seasonal pattern.
    cleaned_mat = np.nan_to_num(mat, nan=fallback_value, posinf=fallback_value, neginf=fallback_value)
    
    # Calculate the average seasonal pattern across the K days.
    # This gives a single representative daily pattern.
    seasonal_pattern = cleaned_mat.mean(axis=0)

    # Tile the averaged daily pattern to cover the entire prediction length.
    num_repetitions = int(np.ceil(prediction_length / season_daily))
    forecast_output = np.tile(seasonal_pattern, num_repetitions)[:prediction_length]

    # --- 4. Final Robustness Check ---
    # As a final safeguard, ensure that no NaN or Inf values have inadvertently
    # appeared in the `forecast_output`. Although `nan_to_num` was applied to the pattern,
    # this provides an extra layer of safety. Replace them with `fallback_value` if any.
    final_forecast = np.nan_to_num(forecast_output, nan=fallback_value, posinf=fallback_value, neginf=fallback_value)
    
    # Return the final forecast as a float array of the specified prediction_length.
    return final_forecast.astype(float)