import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    # If no data, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: Hourly data has strong daily seasonality (period 24).
    season_period = 24

    # Hard requirement 8: Fallback to last-value naive if context is shorter than a full season
    if n < season_period:
        # Hard requirement 4: Never output NaN or inf
        # Hard requirement 9: Indexing safety - context[-1] is safe if n > 0
        return np.full(prediction_length, float(context[-1]), dtype=float)

    # Recommended strategy: Average the last K full days (seasons)
    # K: number of past full seasons to average.
    # We want to average at least 1 full season, and not more than 8 (as per template)
    # and not more than what's available in context.
    K = min(max(1, n // season_period), 8)

    # Hard requirement 9: Indexing safety. Slice the context to get the last K full seasons.
    # The slice context[-K * season_period:] is safe because n >= season_period and K >= 1.
    # So K * season_period <= n.
    recent_seasons_data = context[-K * season_period:]

    # Reshape the data to K rows (days) and `season_period` columns (hours within a day).
    # Then calculate the mean for each hour across the K days.
    # This gives us the average seasonal pattern.
    seasonal_pattern = recent_seasons_data.reshape(K, season_period).mean(axis=0)

    # Tile the seasonal pattern to cover the entire prediction length.
    # Calculate how many repetitions are needed.
    num_repetitions = int(np.ceil(prediction_length / season_period))
    
    # Tile the pattern and then truncate to exactly `prediction_length`.
    forecast_output = np.tile(seasonal_pattern, num_repetitions)[:prediction_length]

    # Hard requirement 4: Never output NaN or inf.
    # Use context[-1] as a robust fallback for any potential NaN values, though unlikely with mean.
    # Hard requirement 9: Indexing safety - context[-1] is safe here because n >= season_period >= 1.
    return np.nan_to_num(forecast_output, nan=float(context[-1])).astype(float)