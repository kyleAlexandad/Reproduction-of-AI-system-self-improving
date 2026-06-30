import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- 1. Determine a robust fallback value ---
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

    # Define the primary daily seasonality for hourly data.
    # The critical benchmark facts strongly indicate period 24 as the dominant and strong seasonality.
    season_daily = 24

    # --- 2. Handle very short contexts ---
    # If the context is empty or shorter than a full daily season,
    # we cannot establish a reliable seasonal pattern. In these cases, return a constant forecast
    # using the robust fallback value. This effectively acts as a last-value naive forecast
    # (or 0.0 if no valid last value) for scenarios with insufficient historical data.
    if n == 0 or n < season_daily:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 3. Calculate Seasonal Pattern by Averaging Last K Daily Seasons ---
    # This strategy improves upon simple seasonal-naive by denoising the pattern
    # by averaging multiple past seasons, making it more robust to recent noise or outliers.

    # K: number of past full seasons to average.
    # Cap K to a reasonable maximum (e.g., 8) to prevent using too distant history,
    # and ensure at least 1 season is used if available.
    max_K_seasons = 8
    K = min(max(1, n // season_daily), max_K_seasons)

    # Extract the relevant historical data for averaging.
    # We take K * season_daily points from the end of the context.
    # This slice is guaranteed to be valid because `n >= season_daily` and `K >= 1` are checked.
    relevant_context = context[-K * season_daily:]

    # Robustify relevant_context by replacing NaNs/Infs with the fallback_value.
    # This ensures that averaging will not produce NaNs or Infs.
    relevant_context_clean = np.nan_to_num(relevant_context, nan=fallback_value)

    # Reshape the data into a matrix where each row is a full season of `season_daily` points.
    # The shape will be (K, season_daily).
    seasonal_matrix = relevant_context_clean.reshape(K, season_daily)

    # Calculate the average across the K seasons for each hour position.
    # This results in a 1D array of length `season_daily`, representing the denoised seasonal pattern.
    seasonal_pattern = seasonal_matrix.mean(axis=0)

    # --- 4. Apply a Level Adjustment ---
    # The averaged seasonal pattern might be at a different overall level than the very latest observations.
    # A simple level adjustment (similar to an ETS level component) can adapt the forecast to recent shifts
    # without introducing a strong trend.

    # Calculate the mean level of the *last observed full season* from the original context.
    # Use np.nanmean to ignore NaNs for this calculation; it returns NaN if all values are NaN.
    last_season_actual_values = context[-season_daily:]
    last_season_mean = np.nanmean(last_season_actual_values)

    # Use a robust estimate for the "current" level:
    # If `last_season_mean` is finite, use it. Otherwise, fall back to the pre-calculated `fallback_value`.
    current_level = last_season_mean if np.isfinite(last_season_mean) else fallback_value

    # Calculate the mean level of the *derived averaged seasonal pattern*.
    # `seasonal_pattern` is already guaranteed finite from step 3.
    pattern_level = np.mean(seasonal_pattern)

    # Calculate the level shift needed to align the pattern's level with the current observed level.
    level_shift = current_level - pattern_level

    # Apply the level shift to the seasonal pattern.
    # The `seasonal_pattern` and `level_shift` are both finite, so `adjusted_seasonal_pattern` will be finite.
    adjusted_seasonal_pattern = seasonal_pattern + level_shift

    # --- 5. Generate Forecasts ---
    # Tile the adjusted seasonal pattern to cover the entire `prediction_length`.
    reps = int(np.ceil(prediction_length / season_daily))
    
    # Tile the pattern and then slice to exactly `prediction_length` values.
    forecast_output = np.tile(adjusted_seasonal_pattern, reps)[:prediction_length]

    # --- 6. Final Robustness Check ---
    # As a final safeguard, ensure that no NaN or Inf values have inadvertently
    # appeared in the `forecast_output`. `nan_to_num` handles both by replacing with `fallback_value`
    # or system min/max if no `nan` argument is provided.
    final_forecast = np.nan_to_num(forecast_output, nan=fallback_value)
    
    # Return the forecast as a float array of the specified prediction_length.
    return final_forecast.astype(float)