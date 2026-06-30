import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    """
    Univariate time-series forecasting for hourly data with strong daily seasonality.

    This function implements a seasonal forecasting approach that averages the last K
    full seasonal periods (daily, period=24 for hourly data) to create a robust seasonal
    pattern, which is then tiled for the forecast horizon. It includes robust handling
    for short context arrays.

    Args:
        context (np.ndarray): A 1D numpy array of the past target values for one series.
        prediction_length (int): The forecast horizon.
        freq (str): A frequency string, e.g., "H".
        metadata (dict, optional): An optional dictionary with series metadata.

    Returns:
        np.ndarray: A 1D numpy array of length `prediction_length` with point forecasts.
    """
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly.
    # Case 1: Empty context. Return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: Hourly data has strong daily seasonality (period 24).
    season_period = 24

    # Hard requirement 8: Handle short context arrays robustly.
    # Case 2: Context shorter than one full seasonal period. Fallback to last-value naive.
    if n < season_period:
        # Hard requirement 4: Never output NaN or inf.
        return np.full(prediction_length, float(context[-1]))

    # Recommended strategy: Average the last K full days (seasonal periods).
    # K is chosen robustly: at least 1, up to the number of available full seasons,
    # and capped at 8 to avoid using very old, potentially stale data.
    K = min(max(1, n // season_period), 8)

    # Hard requirement 9: Indexing safety. Slice from the end.
    # Take the last K full seasonal periods from the context.
    # This covers `K * season_period` points.
    seasonal_data = context[-K * season_period:]

    # Reshape the data into a K x season_period matrix, where each row is a full season.
    # Example: if K=2, season_period=24, data = [day1_hour0, ..., day1_hour23, day2_hour0, ..., day2_hour23]
    # Reshaped:
    # [[day1_hour0, ..., day1_hour23],
    #  [day2_hour0, ..., day2_hour23]]
    seasonal_matrix = seasonal_data.reshape(K, season_period)

    # Average across the K seasons (rows) to get a denoised seasonal pattern for each hour.
    # The result is a 1D array of length `season_period`.
    seasonal_pattern = seasonal_matrix.mean(axis=0)

    # Tile the seasonal pattern to cover the entire prediction_length.
    # Calculate how many repetitions are needed.
    num_reps = int(np.ceil(prediction_length / season_period))
    tiled_forecast = np.tile(seasonal_pattern, num_reps)

    # Hard requirement 3: Output length must be EXACTLY prediction_length.
    forecast_output = tiled_forecast[:prediction_length]

    # Hard requirement 4: Never output NaN or inf. Clip/guard as needed.
    # np.nan_to_num handles potential NaNs if they somehow arise from mean (unlikely with float context).
    # Fallback to the last context value for NaN.
    return np.nan_to_num(forecast_output, nan=float(context[-1])).astype(float)