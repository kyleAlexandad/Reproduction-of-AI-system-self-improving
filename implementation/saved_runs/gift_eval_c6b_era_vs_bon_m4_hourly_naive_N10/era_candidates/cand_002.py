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

    # --- Daily Seasonal Naive component (F_24) ---
    # This serves as the primary, strong baseline (MASE ~1.19).
    if n < season_daily:
        # Fallback to last-value naive if context is shorter than a full daily season.
        # Hard requirement 4 & 9: Safe fallback.
        daily_forecast_base = np.full(prediction_length, float(context[-1]), dtype=float)
    else:
        # Hard requirement 9: Indexing safety. Slice the context to get the last full daily season.
        daily_seasonal_pattern = context[-season_daily:]
        num_repetitions = int(np.ceil(prediction_length / season_daily))
        daily_forecast_base = np.tile(daily_seasonal_pattern, num_repetitions)[:prediction_length]
        
    # --- Weekly Seasonal Naive component (F_168) ---
    # This component is blended in if enough historical data is available.
    if n < season_weekly:
        # If not enough data for a full weekly season, do not use the weekly component.
        weekly_forecast_base = daily_forecast_base # Default to daily forecast
        use_weekly = False
    else:
        # Hard requirement 9: Indexing safety. Slice the context to get the last full weekly season.
        weekly_seasonal_pattern = context[-season_weekly:]
        num_repetitions = int(np.ceil(prediction_length / season_weekly))
        weekly_forecast_base = np.tile(weekly_seasonal_pattern, num_repetitions)[:prediction_length]
        use_weekly = True

    # --- Blending Seasonalities ---
    # The prompt suggests daily seasonality is very strong. Weight it more.
    forecast_output = np.empty(prediction_length, dtype=float)
    if use_weekly:
        # Blend daily and weekly forecasts. Empirically, daily is often stronger for hourly data.
        # Use a weight (e.g., 0.7 for daily, 0.3 for weekly) to prioritize the daily pattern.
        w_daily = 0.7
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
        
        # The drift is the change in average level per daily season.
        level_drift = mean_last_daily_season - mean_prev_daily_season
        
        # Apply a damping factor to reduce the influence of the drift further into the future.
        # A factor close to 1 implies slow damping, closer to 0 implies fast damping.
        damping_factor = 0.95 

        for i in range(prediction_length):
            # The forecast point for index `i` belongs to future season block `i // season_daily`.
            # The drift's contribution is damped exponentially for each successive future season block.
            damped_drift_contribution = level_drift * (damping_factor ** (i // season_daily))
            forecast_output[i] += damped_drift_contribution

    # Hard requirement 4: Never output NaN or inf.
    # np.nan_to_num handles any potential NaNs (e.g., if input context had NaNs and mean resulted in NaN).
    # context[-1] is a safe fallback value because n > 0 is guaranteed by this point.
    return np.nan_to_num(forecast_output, nan=float(context[-1])).astype(float)