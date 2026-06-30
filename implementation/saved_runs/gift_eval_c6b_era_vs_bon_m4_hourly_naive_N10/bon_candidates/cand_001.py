import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    """
    Univariate time-series forecasting for hourly data with strong daily seasonality.

    This function implements a seasonal naive approach by averaging the last K full
    daily cycles (period 24) and repeating this averaged pattern for the forecast horizon.
    It handles short context arrays robustly with appropriate fallbacks.

    Args:
        context (np.ndarray): A 1D numpy array of past target values (history only).
        prediction_length (int): The forecast horizon.
        freq (str): A frequency string, e.g., "H".
        metadata (dict, optional): An optional dictionary with series metadata.
                                   Not used in this implementation, as the
                                   seasonal period is hardcoded based on problem specs.

    Returns:
        np.ndarray: A 1D numpy array of length `prediction_length` with the point forecasts.
    """
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle SHORT context arrays robustly
    # If no history, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical Benchmark Fact: Hourly data has strong daily seasonality (period 24).
    season_period = 24

    # If context is shorter than one season, fall back to last-value naive.
    if n < season_period:
        # Hard requirement 8: Handle short contexts.
        # Hard requirement 4: Never output NaN or inf. context[-1] is guaranteed to exist.
        return np.full(prediction_length, context[-1], dtype=float)

    # Recommended Strategy: Average the last K full days (seasons).
    # K: number of full seasons to average.
    # We use at least 1 full season, and at most 8 to prevent using excessively old data,
    # or more seasons than available in the context.
    # Hard requirement 9: INDEXING SAFETY for K * season
    K = min(max(1, n // season_period), 8)

    # Extract the last K full seasons from the context.
    # The slice context[-K * season_period:] is safe because K * season_period <= n.
    mat = context[-K * season_period:].reshape(K, season_period)

    # Calculate the average pattern for each hour of the day.
    seasonal_pattern = mat.mean(axis=0)

    # Tile the seasonal pattern to cover the prediction length.
    reps = int(np.ceil(prediction_length / season_period))
    forecasts = np.tile(seasonal_pattern, reps)[:prediction_length]

    # Hard requirement 4: Never output NaN or inf.
    # Use context[-1] as a fallback in case seasonal_pattern contains NaNs (e.g., if input context had NaNs).
    # This also ensures the output type is float.
    return np.nan_to_num(forecasts, nan=context[-1]).astype(float)
