import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    # If no data, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: daily seasonality (period 24) for hourly data.
    season = 24

    # Fallback to last-value naive if there isn't enough data for one full season.
    # This covers contexts of length 1 to 23.
    if n < season:
        return np.full(prediction_length, float(context[-1]))

    # --- Improved Seasonal Pattern Generation ---
    # As recommended, average the last K full days (seasons) to denoise the pattern.
    # K: number of seasons to average.
    # K must be at least 1 (if n >= season) and capped at 8 to avoid using excessively old data
    # that might no longer be representative of the current seasonal cycle.
    K = min(max(1, n // season), 8)

    # Extract the data for the last K complete seasons.
    # This slice is safe because K * season <= n (by the definition and calculation of K).
    data_for_seasonal_average = context[-K * season:]

    # Reshape the data to have K rows (each representing one full season) and 'season' columns.
    # Compute the mean across rows to get the averaged seasonal pattern.
    # This denoised pattern is more stable than relying on just the very last season.
    seasonal_pattern = data_for_seasonal_average.reshape(K, season).mean(axis=0)
    # --- End of Improved Seasonal Pattern Generation ---

    # Tile the seasonal pattern to cover the entire prediction length.
    reps = int(np.ceil(prediction_length / season))
    seasonal_tiled_forecast = np.tile(seasonal_pattern, reps)[:prediction_length]

    # Initialize the final forecasts array with the averaged seasonal pattern.
    forecasts = seasonal_tiled_forecast.copy()

    # Add a small, damped level/trend correction, as recommended.
    # This correction requires at least two full seasons of data to estimate a trend.
    # The condition `n >= 2 * season` ensures `context[-1 - season]` is a valid index.
    if n >= 2 * season:
        # Estimate the recent change in level over one seasonal period.
        # This is calculated as the difference between the last observed value
        # and its corresponding value from the previous season. This effectively
        # estimates a seasonal difference, indicating an overall upward or downward trend.
        level_trend_over_season = context[-1] - context[-1 - season]

        # Apply exponential damping to the trend.
        # A damping factor of 0.95 reduces the trend effect by 5% at each future step,
        # preventing over-extrapolation of a potentially temporary trend.
        damping_factor = 0.95

        # Apply the damped level trend to each forecast step.
        # The term (h + 1) ensures the trend is applied starting from the first forecast step (h=0),
        # which is 1 step into the future, and decays progressively.
        for h in range(prediction_length):
            forecasts[h] += level_trend_over_season * (damping_factor ** (h + 1))

    # Hard requirement 4: Never output NaN or inf. Clip/guard as needed.
    # Determine a robust fallback value for `np.nan_to_num`.
    # We assume context[-1] is generally a valid float for this purpose based on problem context.
    # Added explicit handling for positive/negative infinity as well.
    fallback_value = float(context[-1]) if n > 0 else 0.0
    
    return np.nan_to_num(
        forecasts, 
        nan=fallback_value, 
        posinf=np.finfo(float).max, 
        neginf=np.finfo(float).min
    ).astype(float)