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

    # --- 3. Core Forecasting Logic: Averaging last K daily seasons ---
    # The prompt recommends: "average the last K full days at each hour to denoise the pattern."
    # Determine K, the number of daily seasons to average.
    # `n // season_daily` gives the maximum number of full daily seasons available.
    # `min(..., 8)` caps K to 8 seasons (days) to avoid using overly old data and to keep computation lightweight.
    # Since `n >= season_daily` is checked above, `n // season_daily` will be at least 1, so `K` will also be at least 1.
    K = min(n // season_daily, 8) 
        
    # Extract the relevant part of the context: the last K full daily seasons.
    # The length of this segment is `K * season_daily`.
    # This slice is guaranteed to be valid because `n >= K * season_daily` (due to `K <= n // season_daily`).
    seasonal_data_segment = context[-K * season_daily:]
    
    # Robustify the extracted data segment: replace any NaN or Inf values with the `fallback_value`.
    # This ensures that the generated seasonal pattern will not contain any invalid numbers.
    seasonal_data_segment = np.nan_to_num(
        seasonal_data_segment, 
        nan=fallback_value, 
        posinf=fallback_value, 
        neginf=fallback_value
    )

    # Reshape the data segment into a `K x season_daily` matrix.
    # Each row in `mat` represents a full daily season.
    mat = seasonal_data_segment.reshape(K, season_daily)
    
    # Calculate the average pattern across these K seasons.
    # This results in a single array of length `season_daily`, representing the typical daily pattern.
    averaged_seasonal_pattern = mat.mean(axis=0)

    # Tile this averaged daily pattern to cover the entire prediction length.
    # Calculate how many repetitions are needed to cover `prediction_length`.
    num_repetitions = int(np.ceil(prediction_length / season_daily))
    
    # Create the full forecast by tiling the averaged pattern.
    # Then, truncate the tiled array to the exact `prediction_length`.
    forecast_output = np.tile(averaged_seasonal_pattern, num_repetitions)[:prediction_length]

    # --- 4. Final Robustness Check ---
    # As a final safeguard, ensure that no NaN or Inf values have inadvertently
    # appeared in the `forecast_output`. Replace them with `fallback_value` if any.
    final_forecast = np.nan_to_num(
        forecast_output, 
        nan=fallback_value, 
        posinf=fallback_value, 
        neginf=fallback_value
    )
    
    # Return the final forecast as a float array of the specified prediction_length.
    return final_forecast.astype(float)