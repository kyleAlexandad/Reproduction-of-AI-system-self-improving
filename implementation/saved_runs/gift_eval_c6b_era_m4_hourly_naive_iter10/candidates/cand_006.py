import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    # If no data, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: daily seasonality (period 24) for hourly data.
    # The prompt explicitly states that SEASONAL-NAIVE with period 24 is STRONG (MASE ~= 1.19).
    season_daily = 24

    # Fallback to last-value naive if there isn't enough data for one full daily season.
    # This covers contexts of length 1 to 23.
    if n < season_daily:
        return np.full(prediction_length, float(context[-1]))

    # --- Primary Seasonal Pattern: Average the last K full daily cycles ---
    # This approach denoises the seasonal pattern compared to just using the last cycle.
    # K: number of full daily cycles (24-hour periods) to average.
    # Capped at 8 to ensure recentness and prevent using overly old patterns,
    # while ensuring K is at least 1 if n >= season_daily.
    K_daily = min(max(1, n // season_daily), 8)

    # Extract the last K full daily cycles from the context.
    # Indexing safety: K_daily * season_daily will always be <= n,
    # as K_daily is determined by n // season_daily.
    seasonal_data_for_averaging_daily = context[-K_daily * season_daily:]

    # Reshape the data into K_daily rows (each a full 24-hour season) and 'season_daily' columns.
    # Then compute the mean along axis 0 to get the averaged daily seasonal pattern.
    mat_daily = seasonal_data_for_averaging_daily.reshape(K_daily, season_daily)
    averaged_daily_seasonal_pattern = mat_daily.mean(axis=0)

    # Tile the averaged daily seasonal pattern to cover the entire prediction length.
    reps_daily = int(np.ceil(prediction_length / season_daily))
    forecasts = np.tile(averaged_daily_seasonal_pattern, reps_daily)[:prediction_length]

    # --- Optional Secondary Seasonal Pattern: Weekly Seasonality (Period 168) ---
    # The prompt mentions "weekly seasonality (period 168) may also help."
    # We will blend it with the daily pattern if sufficient data is available.
    season_weekly = 168  # 24 hours * 7 days

    if n >= season_weekly:
        # K_weekly: number of full weekly cycles to average.
        # Capped at 4 weeks to maintain recency.
        K_weekly = min(max(1, n // season_weekly), 4)

        seasonal_data_for_averaging_weekly = context[-K_weekly * season_weekly:]
        mat_weekly = seasonal_data_for_averaging_weekly.reshape(K_weekly, season_weekly)
        averaged_weekly_seasonal_pattern = mat_weekly.mean(axis=0)

        reps_weekly = int(np.ceil(prediction_length / season_weekly))
        weekly_forecasts = np.tile(averaged_weekly_seasonal_pattern, reps_weekly)[:prediction_length]

        # Blend the daily and weekly patterns. Given the strong daily seasonality (MASE ~1.19),
        # we assign a higher weight to the daily component. This creates a composite seasonal forecast.
        # This also acts as a robust form of level adjustment by averaging different seasonal levels.
        forecasts = 0.7 * forecasts + 0.3 * weekly_forecasts
    
    # --- NO Damped Level/Trend Correction ---
    # The parent candidate's MASE of ~2.15 was worse than the simple seasonal naive (MASE ~1.19).
    # The prompt explicitly warns: "damped trend (~12.04) are NOT better than naive."
    # Therefore, omitting the damped level/trend correction, which likely hurt performance,
    # is a key improvement to align with benchmark facts and achieve better MASE.

    # Hard requirement 4: Never output NaN or inf. Clip/guard as needed.
    # Replace any potential NaNs or Infs in the forecasts with a sensible fallback value.
    # We use the last valid context value if available, otherwise 0.0.
    fallback_value = float(context[-1]) if n > 0 else 0.0
    return np.nan_to_num(forecasts, nan=fallback_value, posinf=np.finfo(float).max, neginf=np.finfo(float).min).astype(float)