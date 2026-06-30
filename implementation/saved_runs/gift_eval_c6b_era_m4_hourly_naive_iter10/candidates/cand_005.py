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

    # --- Base Seasonal Pattern: Average the last K full daily cycles ---
    # This approach denoises the seasonal pattern compared to just using the last cycle.
    # K: number of full daily cycles to average.
    # Capped at 8 to ensure recentness and prevent using too much memory or very old patterns.
    # It ensures K is at least 1 if n >= season.
    K = min(max(1, n // season), 8)

    # Extract the last K full daily cycles from the context.
    # Indexing safety: Since n >= season, and K is calculated as min(max(1, n // season), 8),
    # K * season will always be less than or equal to n.
    # For example, if n=24, K=1, K*season=24. context[-24:] is safe.
    # If n=40, K=1, K*season=24. context[-24:] is safe.
    # If n=48, K=2, K*season=48. context[-48:] is safe.
    seasonal_data_for_averaging = context[-K * season:]

    # Reshape the data into K rows (each row is a full season) and 'season' columns.
    # Then compute the mean along axis 0 to get the averaged seasonal pattern.
    mat = seasonal_data_for_averaging.reshape(K, season)
    averaged_seasonal_pattern = mat.mean(axis=0)

    # Tile the averaged seasonal pattern to cover the entire prediction length.
    reps = int(np.ceil(prediction_length / season))
    forecasts = np.tile(averaged_seasonal_pattern, reps)[:prediction_length]

    # --- Damped Level/Trend Correction ---
    # This part is adapted from the parent candidate and adds a small, damped level/trend
    # correction on top of the robust seasonal pattern.
    # It requires at least two full seasons of data to estimate a meaningful trend over one season.
    if n >= 2 * season:
        # Estimate the recent change in level over one seasonal period.
        # This is the difference between the last observed value and its corresponding value
        # from the previous season. This captures overall recent drift.
        # Indexing safety: n >= 2 * season ensures context[-1] and context[-1 - season] are valid.
        # (e.g., if n = 2*season, then context[-1-season] is context[season-1], which is valid).
        level_trend_over_season = context[-1] - context[-1 - season]

        # Apply exponential damping to the trend. A damping factor of 0.95 means the
        # trend effect decays by 5% at each future step.
        damping_factor = 0.95

        for h in range(prediction_length):
            # The trend effect is applied cumulatively, damped for each future step.
            forecasts[h] += level_trend_over_season * (damping_factor ** (h + 1))

    # Hard requirement 4: Never output NaN or inf. Clip/guard as needed.
    # Replace any potential NaNs or Infs with a sensible fallback value.
    # We use the last valid context value if available, otherwise 0.0.
    fallback_value = float(context[-1]) if n > 0 else 0.0
    return np.nan_to_num(forecasts, nan=fallback_value, posinf=np.finfo(float).max, neginf=np.finfo(float).min).astype(float)