import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- 1. Determine a robust fallback value ---
    # This value will be used if context is empty, too short, all NaNs/Infs,
    # or to fill NaNs in the derived seasonal pattern.
    fallback_value = 0.0
    if n > 0:
        # Find the last finite value in context. This is the most reliable last known value.
        last_finite_idx = np.where(np.isfinite(context))[0]
        if len(last_finite_idx) > 0:
            fallback_value = context[last_finite_idx[-1]]
    # Ensure fallback_value itself is finite, defaulting to 0.0 if not.
    if not np.isfinite(fallback_value):
        fallback_value = 0.0

    # --- 2. Handle very short contexts ---
    # If the context array is empty, return a constant forecast using the fallback value.
    if n == 0:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 3. Define Seasonalities ---
    # Critical benchmark facts indicate strong daily seasonality (period 24) for hourly data.
    # Weekly seasonality (period 168) is also mentioned as potentially helpful.
    season_daily = 24  # Daily seasonality (24 hours in a day).
    season_weekly = 168 # Weekly seasonality (7 days * 24 hours).

    # If context is shorter than one full daily season, fall back to a naive forecast
    # using the robust fallback_value. This ensures stability.
    if n < season_daily:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 4. Calculate Primary Daily Seasonal Pattern (P=24) ---
    # Average the last K_daily full days to get a robust daily pattern.
    # We average between 1 and 8 full days for robustness against recent noise/outliers.
    # K_daily will be at least 1 due to the `n < season_daily` check above.
    K_daily = min(max(1, n // season_daily), 8)
    
    # Extract the last K_daily * season_daily points. This slice is always valid
    # because n >= season_daily and K_daily >= 1.
    daily_history_slice = context[-K_daily * season_daily:]
    
    # Reshape into (K_daily, season_daily) matrix and calculate the mean for each hour.
    # Use nanmean to handle NaNs in history; returns NaN if all values for an hour are NaN.
    daily_pattern_24 = np.nanmean(daily_history_slice.reshape(K_daily, season_daily), axis=0) 

    # --- 5. Calculate Weekly-Averaged Daily Pattern (from P=168) as a longer-term smooth baseline ---
    # This pattern provides a stable daily shape derived from a longer weekly history.
    weekly_avg_daily_pattern_24 = np.full(season_daily, np.nan, dtype=float) # Initialize with NaNs
    
    # Only calculate if there's enough history for at least one full week.
    if n >= season_weekly:
        # Average the last K_weekly full weeks. Capped at 4 weeks to avoid excessively old data.
        K_weekly = min(max(1, n // season_weekly), 4)
        
        # Extract the last K_weekly * season_weekly points. This slice is always valid.
        weekly_history_slice = context[-K_weekly * season_weekly:]
        
        # Reshape into (K_weekly, season_weekly) matrix and calculate the mean for each hour-of-week.
        weekly_pattern_168 = np.nanmean(weekly_history_slice.reshape(K_weekly, season_weekly), axis=0)

        # If the derived 168-point weekly pattern has any finite values, derive a 24-point daily average from it.
        if np.isfinite(weekly_pattern_168).any():
            # Reshape the 168-point pattern (7 days * 24 hours) into a (7, 24) matrix.
            # Then take the nanmean along axis=0 to get an average for each hour-of-day (0-23).
            weekly_avg_daily_pattern_24 = np.nanmean(weekly_pattern_168.reshape(7, season_daily), axis=0)

    # --- 6. Blend Daily Patterns and Impute NaNs for the Final Seasonal Pattern ---
    # Blend the primary recent daily pattern with the more stable, longer-term weekly-averaged daily pattern.
    # This gives more weight to recent observations while allowing the weekly pattern to smooth out noise.
    # If a component is NaN, the other (if finite) is used. If both are NaN, it remains NaN for now.
    
    final_seasonal_pattern = np.copy(daily_pattern_24) # Start with primary daily pattern

    for i in range(season_daily):
        daily_val = daily_pattern_24[i]
        weekly_avg_val = weekly_avg_daily_pattern_24[i]

        if np.isfinite(daily_val) and np.isfinite(weekly_avg_val):
            # Both are finite, blend them with predefined weights (0.7 for recent daily, 0.3 for weekly-averaged).
            final_seasonal_pattern[i] = 0.7 * daily_val + 0.3 * weekly_avg_val
        elif np.isfinite(weekly_avg_val) and not np.isfinite(daily_val):
            # If the recent daily pattern is NaN for this hour, but the weekly-averaged is finite, use the latter.
            final_seasonal_pattern[i] = weekly_avg_val
        # If daily_val is finite and weekly_avg_val is NaN, final_seasonal_pattern[i] already holds daily_val.
        # If both are NaN, it remains NaN, and will be handled by nan_to_num below.

    # Any remaining NaNs in `final_seasonal_pattern` (e.g., if both sources for an hour were NaN)
    # are filled with the global robust `fallback_value`. This ensures the pattern is fully finite.
    final_seasonal_pattern = np.nan_to_num(final_seasonal_pattern, nan=fallback_value)

    # --- 7. Apply Damped Level Adjustment ---
    # This adjusts the overall level of the seasonal pattern based on the most recent daily context.
    adjusted_final_seasonal_pattern = np.copy(final_seasonal_pattern) 
    
    # Extract the last full daily season from the context. (n >= season_daily already checked).
    last_period_context = context[-season_daily:]
    
    # Calculate the mean of this last period, robustly handling any NaNs in it.
    last_period_average = np.nanmean(last_period_context)
    
    # Calculate the mean of our derived (and now fully finite) seasonal pattern.
    pattern_average = np.mean(final_seasonal_pattern)

    # Only apply a level shift if the average of the last period is a valid, finite number.
    if np.isfinite(last_period_average):
        # The raw difference between the recent context level and the seasonal pattern's baseline level.
        level_difference = last_period_average - pattern_average
        
        # Apply a damping factor to smooth out the level adjustment.
        # This prevents over-reaction to short-term fluctuations or potential outliers.
        damping_factor = 0.7 
        damped_level_shift = level_difference * damping_factor
        
        # Add the damped level shift to the entire seasonal pattern.
        adjusted_final_seasonal_pattern += damped_level_shift

    # --- 8. Generate forecasts ---
    # Tile the cleaned and imputed *adjusted* daily seasonal pattern to cover the entire `prediction_length`.
    reps = int(np.ceil(prediction_length / season_daily))
    tiled_forecast = np.tile(adjusted_final_seasonal_pattern, reps)

    # Truncate the tiled pattern to the exact required `prediction_length`.
    final_forecast = tiled_forecast[:prediction_length]

    # Final robustness check: Ensure no NaN or Inf values are returned,
    # replacing any that might have crept in (e.g., from floating point edge cases) with the fallback value.
    final_forecast = np.nan_to_num(final_forecast, nan=fallback_value)
    
    return final_forecast.astype(float)