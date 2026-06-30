import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    # If no data, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: daily seasonality (period 24) for hourly data.
    # This is the primary seasonality.
    season = 24

    # Fallback to last-value naive if there isn't enough data for one full season.
    # This covers contexts of length 1 to 23.
    if n < season:
        return np.full(prediction_length, float(context[-1]))

    # RECOMMENDED STRATEGY: Use a seasonal backbone by averaging the last K full days
    # to denoise the pattern, as suggested by the good seasonal template.
    # K is capped at 8 to avoid using too much distant (potentially stale) history.
    K = min(max(1, n // season), 8)
    
    # Slice the last K full seasons from the context.
    # This is safe because K * season <= (n // season) * season <= n.
    recent_seasonal_data = context[-K * season:]
    
    # Reshape the data into a K x season matrix.
    # Each row represents a full season (day), and columns represent hours of the day.
    mat = recent_seasonal_data.reshape(K, season)
    
    # Calculate the averaged seasonal pattern by taking the mean across the K seasons.
    # This produces a 1D array of length 'season', representing the typical pattern for each hour.
    seasonal_pattern_avg = mat.mean(axis=0)

    # Tile this averaged seasonal pattern to cover the entire prediction length.
    reps = int(np.ceil(prediction_length / season))
    seasonal_tiled_forecast = np.tile(seasonal_pattern_avg, reps)[:prediction_length]

    # Initialize the final forecasts array with the averaged seasonal pattern.
    forecasts = seasonal_tiled_forecast.copy()

    # Optionally add a small, damped level/trend correction on top of the seasonal pattern.
    # This correction requires at least two full seasons of data to estimate a trend.
    # (i.e., n must be at least 2 * season for context[-1 - season] to be valid).
    if n >= 2 * season:
        # Estimate the recent change in level over one seasonal period.
        # This is calculated as the difference between the last observed value
        # and its corresponding value from the previous season.
        # This captures if the series is trending up or down overall, relative to its seasonal pattern.
        # Indexing safety: context[-1] and context[-1 - season] are safe here because n >= 2 * season.
        level_trend_over_season = context[-1] - context[-1 - season]

        # Apply exponential damping to the trend.
        # A damping factor of 0.95 means the trend effect decays by 5% at each step.
        # This prevents over-extrapolation of a potentially short-term trend.
        damping_factor = 0.95

        # Apply the damped level trend to each forecast step.
        # The trend effect is multiplied by (damping_factor ** (h + 1))
        # because h=0 is the first forecast step, so it represents 1 step into the future.
        for h in range(prediction_length):
            forecasts[h] += level_trend_over_season * (damping_factor ** (h + 1))

    # Hard requirement 4: Never output NaN or inf. Clip/guard as needed.
    # Use np.nan_to_num to replace any potential NaNs (e.g., if context[-1] itself was NaN
    # and was used as a fallback or if intermediate calculations resulted in NaN).
    # The fallback value for NaN is the last known value from the context, or 0.0 if context is empty.
    fallback_value = float(context[-1]) if n > 0 else 0.0
    return np.nan_to_num(forecasts, nan=fallback_value).astype(float)