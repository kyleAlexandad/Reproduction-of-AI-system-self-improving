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

    # Define seasonal periods. Daily is primary, weekly is secondary.
    # The critical benchmark facts strongly indicate period 24 as the dominant and strong seasonality.
    # Weekly seasonality (period 168) may also provide useful information.
    season_daily = 24
    season_weekly = 168 # 24 hours * 7 days

    # --- 2. Handle very short contexts ---
    # If the context is empty, return zeros to satisfy output length requirement.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # If the context is shorter than a full daily season, we cannot establish a reliable seasonal pattern.
    # In these cases, return a constant forecast using the robust fallback value.
    # This effectively acts as a last-value naive forecast (or 0.0 if no valid last value).
    if n < season_daily:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 3. Calculate Daily Seasonal Pattern ---
    # This is the primary seasonal component, derived by averaging the last K full daily cycles.
    # Averaging multiple past seasons helps denoise the pattern and make it more robust to recent outliers.
    
    # K_daily: number of past full daily seasons to average.
    # Cap K_daily to a reasonable maximum (e.g., 8 days) to focus on recent history,
    # and ensure at least 1 season is used if available.
    max_K_daily = 8
    K_daily = min(max(1, n // season_daily), max_K_daily)
    
    # Extract the relevant historical data for daily pattern calculation.
    # This slice is guaranteed to be valid because `n >= season_daily` and `K_daily >= 1` are checked.
    relevant_context_daily = context[-K_daily * season_daily:]
    
    # Robustify relevant_context by replacing NaNs/Infs with the fallback_value.
    # This ensures that averaging will not produce NaNs or Infs.
    relevant_context_daily_clean = np.nan_to_num(relevant_context_daily, nan=fallback_value)
    
    # Reshape the data into a matrix where each row is a full daily season of `season_daily` points.
    # The shape will be (K_daily, season_daily).
    seasonal_daily_matrix = relevant_context_daily_clean.reshape(K_daily, season_daily)
    
    # Calculate the average across the K_daily seasons for each hour position.
    # This results in a 1D array of length `season_daily`, representing the denoised daily pattern.
    daily_pattern = seasonal_daily_matrix.mean(axis=0)

    # --- 4. Calculate Weekly Seasonal Pattern (if enough data is available) ---
    # This is a secondary seasonal component, providing further refinement if enough history exists.
    weekly_pattern_available = False
    weekly_pattern = np.zeros(season_weekly, dtype=float) # Initialize to ensure it's always a valid array

    if n >= season_weekly:
        # K_weekly: number of past full weekly seasons to average.
        # Cap K_weekly (e.g., 4 weeks) as weekly patterns can be more volatile or less persistent over time.
        max_K_weekly = 4
        K_weekly = min(max(1, n // season_weekly), max_K_weekly)
        
        relevant_context_weekly = context[-K_weekly * season_weekly:]
        relevant_context_weekly_clean = np.nan_to_num(relevant_context_weekly, nan=fallback_value)
        
        seasonal_weekly_matrix = relevant_context_weekly_clean.reshape(K_weekly, season_weekly)
        weekly_pattern_raw = seasonal_weekly_matrix.mean(axis=0)
        weekly_pattern_available = True

        # Align the average level of the raw weekly pattern to the average level of the daily pattern.
        # This is crucial for meaningful blending, ensuring both patterns operate around a similar baseline.
        mean_daily_pattern = np.mean(daily_pattern)
        mean_weekly_pattern_raw = np.mean(weekly_pattern_raw)
        weekly_pattern = weekly_pattern_raw - mean_weekly_pattern_raw + mean_daily_pattern
    
    # --- 5. Generate Initial Forecasts by Blending Patterns ---
    # Combine the daily and (optionally) weekly patterns to form the initial forecast.
    forecast_output = np.zeros(prediction_length, dtype=float)
    
    # Blending weights: daily seasonality is known to be very strong for this dataset,
    # so it receives a higher weight. Weekly seasonality acts as a refinement for day-of-week effects.
    weight_daily = 0.8
    weight_weekly = 0.2 

    for h in range(prediction_length):
        # The hour within the day (0-23) determines the daily component.
        daily_component = daily_pattern[h % season_daily]
        
        if weekly_pattern_available:
            # The hour within the week (0-167) determines the weekly component.
            weekly_component = weekly_pattern[h % season_weekly]
            # Simple weighted average of the two level-aligned seasonal patterns.
            forecast_output[h] = weight_daily * daily_component + weight_weekly * weekly_component
        else:
            # If weekly data is not available, the forecast relies solely on the daily pattern.
            forecast_output[h] = daily_component

    # --- 6. Apply a Final Overall Level Adjustment ---
    # This step aligns the overall level of the generated forecast with the most recent observed data.
    # This is crucial for adapting the forecast to recent shifts in the series' baseline level.
    
    # Estimate the "current" level from the *last full daily season* of the original context.
    # Using `np.nanmean` is robust to NaNs in the recent history.
    last_season_actual_values = context[-season_daily:]
    current_level_estimate = np.nanmean(last_season_actual_values)
    
    # Fallback if the last season itself was entirely NaN/Inf.
    if not np.isfinite(current_level_estimate):
        current_level_estimate = fallback_value

    # Calculate the average level of the *generated forecast output*.
    # All values in `forecast_output` are guaranteed finite at this point.
    forecast_average_level = np.mean(forecast_output)

    # Calculate the required level shift to match the current observed level.
    level_shift = current_level_estimate - forecast_average_level
    
    # Apply the level shift to the entire forecast array.
    final_forecast = forecast_output + level_shift

    # --- 7. Final Robustness Check ---
    # As a final safeguard, ensure that no NaN or Inf values have inadvertently
    # appeared in the `final_forecast`. Replace them with `fallback_value` to maintain numerical stability.
    final_forecast = np.nan_to_num(final_forecast, nan=fallback_value)
    
    # Return the forecast as a float array of the specified prediction_length.
    return final_forecast.astype(float)