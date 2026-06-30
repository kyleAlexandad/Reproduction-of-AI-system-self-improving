import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    """
    Univariate time-series forecasting for hourly data with strong daily seasonality.

    Leverages seasonal averaging of the last K full days to produce forecasts,
    falling back to last-value naive for very short context histories.

    Args:
        context (np.array): A 1D numpy array of past target values.
        prediction_length (int): The number of steps to forecast.
        freq (str): The frequency of the time series (e.g., "H").
        metadata (dict, optional): Optional metadata (item_id, season_length, etc.).

    Returns:
        np.array: A 1D numpy array of length `prediction_length` with point forecasts.
    """
    context = np.asarray(context, dtype=float)
    n = len(context)

    # 1. Handle empty context: return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # For hourly data, the strongest seasonality is typically daily (24 hours).
    season = 24

    # 2. Handle context shorter than one full season: fall back to last-value naive.
    if n < season:
        return np.full(prediction_length, float(context[-1]), dtype=float)

    # Determine how many full seasons are available in the context.
    num_full_seasons_available = n // season

    # 3. Determine 'K': the number of recent full seasons to average.
    #    Use at least 1, up to a maximum (e.g., 8) to denoise the pattern
    #    without using potentially stale data from very long histories.
    K = min(max(1, num_full_seasons_available), 8)

    # 4. Extract the relevant historical data: the last K full seasons.
    #    This ensures we always have K * season points for reshaping.
    historical_seasons_data = context[-K * season:]

    # 5. Reshape the data into K rows (each row is a season) and 'season' columns
    #    (each column represents a specific hour within the daily cycle).
    mat = historical_seasons_data.reshape(K, season)

    # 6. Calculate the average for each hour of the day across the K seasons.
    #    This produces a denoised 1D seasonal pattern of length 'season'.
    seasonal_pattern = mat.mean(axis=0)

    # 7. Determine how many repetitions of the seasonal pattern are needed
    #    to cover the entire prediction_length.
    reps = int(np.ceil(prediction_length / season))

    # 8. Tile the seasonal pattern and then slice to the exact prediction_length.
    out = np.tile(seasonal_pattern, reps)[:prediction_length]

    # 9. Robustness: ensure no NaNs or Infs are returned and output type is float.
    #    If any NaN somehow occurs (e.g., if input context was full of NaNs),
    #    replace it with the last valid context value.
    return np.nan_to_num(out, nan=float(context[-1]), posinf=np.finfo(float).max, neginf=np.finfo(float).min).astype(float)