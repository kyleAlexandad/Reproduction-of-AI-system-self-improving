import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    """
    Univariate time-series forecasting for hourly data with strong daily seasonality.

    Leverages daily seasonality (period 24) by averaging the last K full days
    and applying a simple level adjustment based on the most recent full season.
    Handles short contexts robustly with appropriate fallbacks.

    Args:
        context (np.ndarray): 1D numpy array of past target values.
        prediction_length (int): The forecast horizon.
        freq (str): Frequency string, e.g., "H".
        metadata (dict, optional): Optional dictionary with series metadata.

    Returns:
        np.ndarray: 1D numpy array of length `prediction_length` with point forecasts.
    """
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle SHORT context arrays robustly
    # If no context is available, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical Benchmark Fact: Hourly data has strong DAILY seasonality (period 24).
    season = 24

    # Hard requirement 8: Fallback to last-value naive if not enough context
    # to form a full seasonal pattern.
    if n < season:
        # Hard requirement 4: Never output NaN or inf.
        return np.full(prediction_length, float(context[-1]))

    # Determine K: number of full seasons to average for the seasonal pattern.
    # Max 8 to keep it local and prevent averaging too much history if context is very long.
    # Min 1 to ensure at least one full season is used if available.
    K = min(max(1, n // season), 8)

    # Hard requirement 9: Indexing safety.
    # Extract the last K full seasons from the context.
    # context[-K * season:] is safe because K * season <= n.
    historical_seasons = context[-K * season:].reshape(K, season)
    
    # Calculate the average seasonal pattern.
    seasonal_pattern = historical_seasons.mean(axis=0)

    # Level adjustment: Adjust the seasonal pattern based on the most recent level.
    # Compare the mean of the last complete season to the mean of the averaged seasonal pattern.
    # This helps in handling recent shifts in the overall level or slow trends.
    # context[-season:] is safe because n >= season at this point.
    last_season_mean = context[-season:].mean()
    averaged_seasonal_pattern_mean = seasonal_pattern.mean()
    level_offset = last_season_mean - averaged_seasonal_pattern_mean

    # Apply the level offset to the seasonal pattern.
    adjusted_seasonal_pattern = seasonal_pattern + level_offset

    # Tile the adjusted seasonal pattern to cover the prediction_length.
    reps = int(np.ceil(prediction_length / season))
    
    # Hard requirement 3: Output length must be EXACTLY prediction_length.
    forecast_values = np.tile(adjusted_seasonal_pattern, reps)[:prediction_length]

    # Hard requirement 4: Never output NaN or inf. Use context[-1] as a robust fallback.
    # This handles potential NaNs if the mean calculation resulted in NaNs (e.g., if
    # all values in a column of `historical_seasons` were NaN, though unlikely with typical data).
    # Ensure the output type is float.
    return np.nan_to_num(forecast_values, nan=float(context[-1])).astype(float)