import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: Hourly data has strong daily seasonality (period 24).
    season_daily = 24

    # Initialize forecast_values. This will be overwritten by fallbacks or the main logic.
    forecast_values = np.zeros(prediction_length, dtype=float)

    # --- Fallbacks and Core Seasonal Forecast ---
    if n < season_daily:
        # Fallback 1: Last-value naive if context is shorter than a full daily season.
        # This handles cases where seasonality cannot be extracted.
        forecast_values = np.full(prediction_length, float(context[-1]), dtype=float)
    else:
        # Core forecast: Simple Seasonal Naive using the last full daily season.
        # The prompt indicates this approach (MASE ~1.19) is strong for this dataset,
        # often outperforming K-averaged seasonal patterns.
        # Hard requirement 9: Indexing safety. context[-season_daily:] is safe because n >= season_daily.
        seasonal_pattern = context[-season_daily:]
        
        # Tile the seasonal pattern to cover the entire prediction_length.
        num_repetitions = int(np.ceil(prediction_length / season_daily))
        forecast_values = np.tile(seasonal_pattern, num_repetitions)[:prediction_length]

        # --- Damped Level Correction ---
        # Apply a damped level correction only if enough history is available (at least 2 full seasons).
        # This aims to capture recent shifts in the overall level of the series, on top of the
        # repeating seasonal pattern, without introducing a problematic linear trend.
        if n >= 2 * season_daily:
            # Calculate the average level of the most recent full season.
            # Hard requirement 9: Indexing safety. context[-season_daily:] is safe.
            mean_last_season = np.mean(context[-season_daily:])
            
            # Calculate the average level of the season immediately preceding the last one.
            # Hard requirement 9: Indexing safety. context[-2*season_daily:-season_daily] is safe.
            mean_season_before_last = np.mean(context[-2*season_daily:-season_daily])
            
            # The 'level_change' represents the observed shift in the average level
            # between the last two full seasons.
            level_change = mean_last_season - mean_season_before_last
            
            # Damping factor. Controls how quickly the influence of the 'level_change' diminishes
            # over the forecast horizon. A value between 0 and 1.
            # A moderate damping (e.g., 0.85) is used to prevent over-extrapolating trends
            # that might not persist far into the future.
            damping_factor = 0.85 

            # Apply the damped level correction to each point in the forecast.
            for i in range(prediction_length):
                # The exponent (i // season_daily) determines which 'season block' the forecast point falls into.
                # For i in [0, 23], exponent is 0: correction is `level_change * (0.85^0) = level_change`.
                # For i in [24, 47], exponent is 1: correction is `level_change * (0.85^1)`.
                # This extrapolates the observed 'seasonal mean trend' into the future, with progressive damping.
                forecast_values[i] += level_change * (damping_factor ** (i // season_daily))

    # Hard requirement 4: Never output NaN or inf.
    # np.nan_to_num replaces any NaNs (e.g., if context contained NaNs that propagated through mean calculations)
    # with a safe fallback value (the last observed value from context).
    # Finally, ensure the output type is float.
    return np.nan_to_num(forecast_values, nan=float(context[-1])).astype(float)
