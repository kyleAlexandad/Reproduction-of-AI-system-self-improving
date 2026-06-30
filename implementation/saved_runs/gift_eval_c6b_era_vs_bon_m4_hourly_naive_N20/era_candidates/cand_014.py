import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- 1. Determine a robust fallback value ---
    # This value is crucial for handling empty, very short, or entirely NaN/Inf contexts.
    # It attempts to use the last known finite value from the context, otherwise defaults to 0.0.
    fallback_value = 0.0
    if n > 0:
        # Find all finite values in the context.
        finite_values = context[np.isfinite(context)]
        if len(finite_values) > 0:
            # If finite values exist, the last one is the most reliable recent value.
            fallback_value = finite_values[-1]
    # Ensure fallback_value itself is finite.
    if not np.isfinite(fallback_value):
        fallback_value = 0.0

    # Define the primary daily seasonality for hourly data.
    season_daily = 24

    # --- 2. Handle very short contexts ---
    # If the context is empty or shorter than a full daily season,
    # we cannot establish a reliable seasonal pattern. In these cases, return a constant forecast
    # using the robust fallback value. This acts as a last-value naive forecast for insufficient data.
    if n == 0 or n < season_daily:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 3. Calculate Averaged Daily Seasonal Pattern ---
    # Average the last K full daily seasons to denoise the pattern.
    # K is chosen to be at least 1 and at most 4 (a balance for robustness and recency for "short" series).
    K_daily = min(max(1, n // season_daily), 4)
    
    # Extract the last K_daily * season_daily observations. This slice is guaranteed to be valid
    # because `n >= season_daily` (and thus `n >= K_daily * season_daily` if K_daily=1) has been checked.
    context_for_seasonal_avg = context[-K_daily * season_daily:]
    
    # Reshape into a matrix where each row represents a full season.
    matrix = context_for_seasonal_avg.reshape(K_daily, season_daily)
    
    # Calculate the mean for each hour of the day across the K_daily seasons, ignoring NaNs.
    # This results in a 24-point daily seasonal pattern.
    daily_seasonal_pattern = np.nanmean(matrix, axis=0)

    # Replace any NaNs that might result from `np.nanmean` (e.g., if an entire column was NaN)
    # with the robust fallback value.
    daily_seasonal_pattern = np.nan_to_num(daily_seasonal_pattern, nan=fallback_value)

    # --- 4. Calculate Damped Level Adjustment ---
    # We want to adjust the seasonal pattern based on the most recent deviation from seasonality.
    # Get the last valid observation value. If context[-1] is NaN, use the last finite value found.
    last_actual_value = context[-1]
    if not np.isfinite(last_actual_value):
        # This branch ensures last_actual_value is finite, using the already computed finite_values.
        if len(finite_values) > 0:
            last_actual_value = finite_values[-1]
        else: # Should not be reached if n > 0 and fallback_value logic is sound
            last_actual_value = fallback_value

    # Determine which point in the daily cycle the last observation corresponds to.
    index_in_season = (n - 1) % season_daily
    # Get the seasonal expectation for that point from our derived pattern.
    seasonal_expected_value_at_last_point = daily_seasonal_pattern[index_in_season]

    # Calculate the difference: how much the last actual value deviated from its seasonal expectation.
    level_error = last_actual_value - seasonal_expected_value_at_last_point

    # Apply a damping factor to this level error over the forecast horizon.
    # A damping factor (e.g., 0.95, 0.98) causes the level adjustment to diminish over time,
    # assuming the deviation might be temporary rather than a permanent shift.
    damping_factor = 0.95
    damped_level_corrections = np.array([level_error * (damping_factor ** h) 
                                         for h in range(prediction_length)])
    
    # --- 5. Generate Base Seasonal Forecast ---
    # Tile the 24-point daily seasonal pattern to cover the entire `prediction_length`.
    reps = int(np.ceil(prediction_length / season_daily))
    base_seasonal_forecast = np.tile(daily_seasonal_pattern, reps)[:prediction_length]

    # --- 6. Combine Base Forecast with Damped Level Adjustment ---
    forecast_output = base_seasonal_forecast + damped_level_corrections

    # --- 7. Final Robustness Check ---
    # Ensure no NaN or Inf values remain. Replace any with the fallback_value.
    final_forecast = np.nan_to_num(forecast_output, nan=fallback_value)
    
    # Return the forecast as a float array of the specified prediction_length.
    return final_forecast.astype(float)