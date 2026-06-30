import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Determine a robust fallback value in case of NaNs later in calculations or in very short context.
    # Prefer the last non-NaN value from context; if all context is NaN, use 0.0.
    non_nan_context = context[~np.isnan(context)]
    if len(non_nan_context) > 0:
        fallback_value = float(non_nan_context[-1])
    else:
        # All context values are NaN. Return 0.0 as a safe numeric fallback.
        fallback_value = 0.0 

    # Critical benchmark fact: Hourly data has strong daily seasonality (period 24).
    # Weekly seasonality (period 168) may also help.
    season_daily = 24
    season_weekly = 168 # 24 hours * 7 days

    # --- Daily Seasonal Pattern (averaged over K past days) ---
    # This serves as the primary, strong baseline. Averaging multiple seasons helps denoise.
    daily_pattern_length = season_daily
    if n < daily_pattern_length:
        # Fallback: if context is shorter than a full daily season, repeat the fallback value
        # as the seasonal pattern.
        daily_pattern = np.full(daily_pattern_length, fallback_value, dtype=float)
    else:
        # Average the last K full daily seasons to create a robust daily pattern.
        # K is limited to a maximum of 8 days to prevent using very old data if context is very long.
        K_daily = min(max(1, n // daily_pattern_length), 8)
        
        # Hard requirement 9: Indexing safety. Slice the context for the relevant history.
        daily_data_for_pattern = context[-K_daily * daily_pattern_length:]
        
        # Reshape the data into K rows (days) and 'season_daily' columns (hours of the day)
        # and compute the mean for each hour.
        # Use np.nan_to_num to handle potential NaNs in the calculated pattern if context contained NaNs.
        daily_pattern = np.nan_to_num(
            daily_data_for_pattern.reshape(K_daily, daily_pattern_length).mean(axis=0),
            nan=fallback_value
        )
    
    # Generate the base forecast by tiling the daily pattern.
    num_repetitions_daily = int(np.ceil(prediction_length / daily_pattern_length))
    daily_forecast_base = np.tile(daily_pattern, num_repetitions_daily)[:prediction_length]

    # --- Weekly Seasonal Pattern (averaged over K past weeks) ---
    # This component is blended in if enough historical data is available.
    weekly_pattern_length = season_weekly
    weekly_forecast_base = daily_forecast_base # Initialize with daily forecast as default
    use_weekly = False

    if n >= weekly_pattern_length:
        # Average the last K full weekly seasons for a robust weekly pattern.
        # K is limited to a maximum of 4 weeks.
        K_weekly = min(max(1, n // weekly_pattern_length), 4)
        
        # Hard requirement 9: Indexing safety.
        weekly_data_for_pattern = context[-K_weekly * weekly_pattern_length:]
        
        weekly_pattern = np.nan_to_num(
            weekly_data_for_pattern.reshape(K_weekly, weekly_pattern_length).mean(axis=0),
            nan=fallback_value
        )
        # Generate the base forecast by tiling the weekly pattern.
        num_repetitions_weekly = int(np.ceil(prediction_length / weekly_pattern_length))
        weekly_forecast_base = np.tile(weekly_pattern, num_repetitions_weekly)[:prediction_length]
        use_weekly = True

    # --- Blending Seasonalities ---
    # The prompt emphasizes daily seasonality is very strong for hourly data.
    # Blend daily and weekly forecasts, giving more weight to the daily pattern.
    forecast_output = np.empty(prediction_length, dtype=float)
    if use_weekly:
        w_daily = 0.7 # Weight for the daily component
        forecast_output = w_daily * daily_forecast_base + (1 - w_daily) * weekly_forecast_base
    else:
        # If weekly data is not sufficient, rely solely on the daily forecast.
        forecast_output = daily_forecast_base

    # --- Damped Level Correction ---
    # Apply a level adjustment to anchor the forecast to the end of the history.
    # This corrects for any immediate deviation of the last observed value from its seasonal expectation.
    # Only apply if we have enough data to determine a meaningful daily pattern and the last value is not NaN.
    if n >= daily_pattern_length and not np.isnan(context[-1]):
        # Determine the seasonal expectation for the last observed point.
        last_idx_in_daily_season = (n - 1) % daily_pattern_length
        current_seasonal_value = daily_pattern[last_idx_in_daily_season]
        
        # Calculate the residual: actual last value minus its seasonal expectation.
        level_residual = context[-1] - current_seasonal_value
        
        # Apply a damping factor to gradually reduce the influence of this residual over the forecast horizon.
        # Damping is applied per daily seasonal block.
        damping_factor_level = 0.9 

        for i in range(prediction_length):
            # The contribution of the level residual decays exponentially for each
            # successive future daily season block.
            damped_level_contribution = level_residual * (damping_factor_level ** (i // daily_pattern_length))
            forecast_output[i] += damped_level_contribution

    # Hard requirement 4: Never output NaN or inf.
    # np.nan_to_num handles any potential NaNs (e.g., if context had NaNs and mean resulted in NaN,
    # or if the level correction introduced NaNs).
    # Use the robust fallback_value determined at the start.
    return np.nan_to_num(forecast_output, nan=fallback_value).astype(float)