import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    # If no data, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: Hourly data has strong daily seasonality (period 24).
    # Weekly seasonality (period 168) may also help.
    season_daily = 24
    season_weekly = 168 # 24 hours * 7 days

    # --- Daily Seasonal Component (F_24) ---
    # This serves as the primary, strong baseline (MASE ~1.19).
    # We improve this by averaging multiple past daily seasons to denoise the pattern.
    if n < season_daily:
        # Fallback to last-value naive if context is shorter than a full daily season.
        daily_forecast_base = np.full(prediction_length, float(context[-1]), dtype=float)
    else:
        # Determine how many full daily seasons to average. Max 8 as per template suggestion.
        # n // season_daily gives the number of available full seasons.
        K_daily = min(max(1, n // season_daily), 8)
        
        # Hard requirement 9: Indexing safety. Slice the last K_daily * season_daily points.
        # This gives us K_daily complete daily cycles from the end of the context.
        daily_seasonal_data = context[-K_daily * season_daily:]
        
        # Reshape the data into (K_daily, season_daily) and average across days (axis=0)
        # to get a single, denoised daily seasonal pattern.
        daily_seasonal_pattern = daily_seasonal_data.reshape(K_daily, season_daily).mean(axis=0)
        
        # Tile this pattern to cover the prediction_length.
        num_repetitions = int(np.ceil(prediction_length / season_daily))
        daily_forecast_base = np.tile(daily_seasonal_pattern, num_repetitions)[:prediction_length]
        
    # --- Weekly Seasonal Component (F_168) ---
    # This component is blended in if enough historical data is available.
    # We also average multiple past weekly seasons here for robustness.
    if n < season_weekly:
        # If not enough data for a full weekly season, do not use the weekly component.
        weekly_forecast_base = daily_forecast_base # Default to daily forecast
        use_weekly = False
    else:
        # Determine how many full weekly seasons to average. Capped at 4 for weekly patterns
        # to ensure sufficient data without requiring excessively long history.
        K_weekly = min(max(1, n // season_weekly), 4) 
        
        # Hard requirement 9: Indexing safety. Slice the last K_weekly * season_weekly points.
        weekly_seasonal_data = context[-K_weekly * season_weekly:]
        
        # Reshape and average across weeks to get the denoised weekly seasonal pattern.
        weekly_seasonal_pattern = weekly_seasonal_data.reshape(K_weekly, season_weekly).mean(axis=0)
        
        # Tile this pattern to cover the prediction_length.
        num_repetitions = int(np.ceil(prediction_length / season_weekly))
        weekly_forecast_base = np.tile(weekly_seasonal_pattern, num_repetitions)[:prediction_length]
        use_weekly = True

    # --- Blending Seasonalities ---
    # The prompt suggests daily seasonality is very strong for hourly data.
    forecast_output = np.empty(prediction_length, dtype=float)
    if use_weekly:
        # Blend daily and weekly forecasts. Prioritize the daily pattern with a higher weight.
        w_daily = 0.7 # Weight for daily component
        forecast_output = w_daily * daily_forecast_base + (1 - w_daily) * weekly_forecast_base
    else:
        # If weekly data is not sufficient, rely solely on the daily forecast.
        forecast_output = daily_forecast_base

    # --- Damped Level/Trend Correction ---
    # "Optionally add a small, damped level/trend correction on top of the seasonal pattern."
    # We estimate a level drift based on the average change between the last two daily seasons.
    level_drift = 0.0
    min_context_for_drift = 2 * season_daily # Need at least two full daily seasons to estimate drift.

    if n >= min_context_for_drift:
        # Calculate the mean of the last complete daily season.
        mean_last_daily_season = np.mean(context[-season_daily:])
        # Calculate the mean of the daily season before the last one.
        # Hard requirement 9: Indexing safety.
        mean_prev_daily_season = np.mean(context[-min_context_for_drift : -season_daily])
        
        # The drift is the change in average level from the previous daily season to the last.
        level_drift = mean_last_daily_season - mean_prev_daily_season
        
        # Apply a damping factor to reduce the influence of the drift further into the future.
        damping_factor = 0.95 

        for i in range(prediction_length):
            # The drift's contribution is damped exponentially for each successive future daily season block.
            # i // season_daily gives the season block index (0 for first 24 points, 1 for next 24, etc.)
            damped_drift_contribution = level_drift * (damping_factor ** (i // season_daily))
            forecast_output[i] += damped_drift_contribution

    # Hard requirement 4: Never output NaN or inf.
    # np.nan_to_num handles any potential NaNs (e.g., if input context had NaNs and mean resulted in NaN).
    # context[-1] is a safe fallback value because n > 0 is guaranteed by the initial check.
    return np.nan_to_num(forecast_output, nan=float(context[-1])).astype(float)