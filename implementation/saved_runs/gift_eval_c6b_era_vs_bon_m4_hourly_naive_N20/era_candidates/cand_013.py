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

    # Define common seasonal periods for hourly data
    season_daily = 24  # Daily seasonality (24 hours)
    season_weekly = 168 # Weekly seasonality (7 days * 24 hours)

    # --- 2. Handle very short contexts ---
    # If the context is empty or shorter than a full daily season,
    # we cannot establish a reliable seasonal pattern. In these cases, return a constant forecast
    # using the robust fallback value. This acts as a last-value naive forecast.
    if n == 0 or n < season_daily:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 3. Extract and Denoise Daily Seasonal Pattern (Period 24) ---
    # This averages the last K full daily seasons to create a more robust pattern
    # than just using the last single day.
    # K_daily: number of full daily seasons (24 hours) to average.
    # It is capped at 8 to prevent using excessively stale history.
    K_daily = min(max(1, n // season_daily), 8)

    # Extract the relevant history and reshape into a matrix where each row is a full day.
    # The slice `context[-K_daily * season_daily:]` is guaranteed to be valid as `K_daily * season_daily <= n`.
    matrix_daily = context[-K_daily * season_daily:].reshape(K_daily, season_daily)
    
    # Calculate the average for each hour of the day across these K_daily days.
    daily_seasonal_pattern = matrix_daily.mean(axis=0)

    # Robustify the averaged pattern: replace any NaN/Inf values that might result
    # (e.g., if an entire column in `matrix_daily` was NaN).
    daily_seasonal_pattern = np.nan_to_num(daily_seasonal_pattern, nan=fallback_value)

    # --- 4. Optionally incorporate Weekly Seasonal Pattern (Period 168) ---
    # This step is conditional: it only applies if there is enough historical data
    # (at least one full week) to derive a meaningful weekly pattern.
    
    forecast_pattern_to_tile = daily_seasonal_pattern # Default pattern is daily
    pattern_tile_length = season_daily # Default tiling period is daily

    if n >= season_weekly:
        # If enough data for at least one full week, calculate and blend a weekly pattern.
        # K_weekly: number of full weekly seasons (168 hours) to average.
        # Capped at 4 weeks to balance stability and recency.
        K_weekly = min(max(1, n // season_weekly), 4)
        
        # Extract and reshape history for weekly pattern extraction.
        matrix_weekly = context[-K_weekly * season_weekly:].reshape(K_weekly, season_weekly)
        weekly_seasonal_pattern = matrix_weekly.mean(axis=0)
        weekly_seasonal_pattern = np.nan_to_num(weekly_seasonal_pattern, nan=fallback_value)

        # --- Blend the daily and weekly patterns ---
        # A simple additive blend to combine the short-term (daily) and
        # longer-term (weekly) seasonal components.
        # The `alpha_weekly_blend` controls the weight given to the weekly pattern.
        # A lower weight is chosen as daily seasonality is typically stronger for hourly data.
        alpha_weekly_blend = 0.25 # 25% weight for weekly, 75% for daily

        # To blend, the daily pattern needs to be expanded to a full week's length.
        daily_pattern_projected_to_week = np.tile(daily_seasonal_pattern, 7)[:season_weekly]
        
        # The blended pattern for one full week.
        # This combines the general weekly shape from `weekly_seasonal_pattern`
        # with the more recent, denoised `daily_seasonal_pattern`.
        blended_weekly_pattern = (1 - alpha_weekly_blend) * daily_pattern_projected_to_week + \
                                 alpha_weekly_blend * weekly_seasonal_pattern
        
        forecast_pattern_to_tile = blended_weekly_pattern
        pattern_tile_length = season_weekly # Now tile the 168-hour blended pattern

    # --- 5. Generate Forecasts from the chosen (daily or blended weekly) pattern ---
    # Calculate how many repetitions of the chosen pattern are needed to cover the prediction_length.
    reps = int(np.ceil(prediction_length / pattern_tile_length))
    
    # Tile the pattern and then slice to obtain exactly `prediction_length` values.
    forecast_output = np.tile(forecast_pattern_to_tile, reps)[:prediction_length]

    # --- 6. Final Robustness Check ---
    # As a final safeguard, ensure that no NaN or Inf values have inadvertently
    # appeared in the `forecast_output`. Replace any such values with the `fallback_value`.
    final_forecast = np.nan_to_num(forecast_output, nan=fallback_value)
    
    # Return the forecast as a float array of the specified prediction_length.
    return final_forecast.astype(float)