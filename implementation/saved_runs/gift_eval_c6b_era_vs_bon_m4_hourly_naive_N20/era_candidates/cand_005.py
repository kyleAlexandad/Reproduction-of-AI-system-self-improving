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

    # --- 4. Calculate Daily Seasonal Pattern (P=24) ---
    # Average the last K full days to get a robust daily pattern.
    # We average between 1 and 8 full days for robustness against recent noise/outliers.
    K_daily = min(max(1, n // season_daily), 8)
    
    # Extract the last K_daily * season_daily points. This slice is always valid.
    daily_history_slice = context[-K_daily * season_daily:]
    
    # Reshape into (K_daily, season_daily) matrix and calculate the mean for each hour.
    daily_mat = daily_history_slice.reshape(K_daily, season_daily)
    daily_pattern_candidate = np.nanmean(daily_mat, axis=0) # Use nanmean to handle NaNs in history

    # --- 5. Calculate Weekly Seasonal Pattern (P=168) as a backup/imputation source ---
    weekly_pattern_candidate = None
    # Only calculate if there's enough history for at least one full week.
    if n >= season_weekly:
        # Average the last K full weeks. Capped at 4 weeks to avoid excessively old data.
        K_weekly = min(max(1, n // season_weekly), 4)
        
        # Extract the last K_weekly * season_weekly points. This slice is always valid.
        weekly_history_slice = context[-K_weekly * season_weekly:]
        
        # Reshape into (K_weekly, season_weekly) matrix and calculate the mean for each hour-of-week.
        weekly_mat = weekly_history_slice.reshape(K_weekly, season_weekly)
        weekly_pattern_candidate = np.nanmean(weekly_mat, axis=0)

    # --- 6. Final Seasonal Pattern Construction and Imputation ---
    # The primary pattern is the daily seasonal pattern.
    final_seasonal_pattern = np.copy(daily_pattern_candidate)

    # Identify any NaN values in the primary daily pattern.
    nan_indices_in_daily = np.where(np.isnan(final_seasonal_pattern))[0]

    if len(nan_indices_in_daily) > 0 and weekly_pattern_candidate is not None:
        # If there are NaNs in the daily pattern (meaning all K_daily values for those hours were NaN),
        # try to fill them using information from the weekly pattern.
        # This leverages longer-term seasonality to fill gaps in the shorter-term pattern.
        
        # Ensure weekly_pattern_candidate actually contains finite values before using it.
        if np.isfinite(weekly_pattern_candidate).any():
            # Calculate a robust average for each hour-of-day from the weekly pattern.
            # This provides a long-term daily pattern, averaged across all days of the week in the weekly history.
            # Reshape weekly_pattern_candidate (length 168) into (number_of_days_in_weekly_history, 24).
            weekly_pattern_reshaped_for_daily_avg = weekly_pattern_candidate.reshape(-1, season_daily)
            
            # Take nanmean along axis=0 to get an average for each hour-of-day (0-23).
            long_term_daily_avg_from_weekly = np.nanmean(weekly_pattern_reshaped_for_daily_avg, axis=0)

            # Fill NaNs in final_seasonal_pattern using these historical hourly averages from weekly data.
            for idx in nan_indices_in_daily:
                if np.isfinite(long_term_daily_avg_from_weekly[idx]):
                    final_seasonal_pattern[idx] = long_term_daily_avg_from_weekly[idx]
    
    # Any remaining NaNs in `final_seasonal_pattern` (e.g., if no weekly pattern was available,
    # or all values for a specific hour were NaN even across weeks) are filled with the
    # global robust `fallback_value`.
    final_seasonal_pattern = np.nan_to_num(final_seasonal_pattern, nan=fallback_value)

    # --- 7. Generate forecasts ---
    # Tile the cleaned and imputed daily seasonal pattern to cover the entire `prediction_length`.
    reps = int(np.ceil(prediction_length / season_daily))
    tiled_forecast = np.tile(final_seasonal_pattern, reps)

    # Truncate the tiled pattern to the exact required `prediction_length`.
    final_forecast = tiled_forecast[:prediction_length]

    # Ensure the final output array has a float data type.
    return final_forecast.astype(float)