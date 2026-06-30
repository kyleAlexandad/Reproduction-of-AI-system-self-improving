import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    """
    Univariate time-series forecasting for hourly data with strong daily seasonality (period 24).
    The function forecasts the next `prediction_length` steps using a seasonal pattern
    derived from the last K full days, optionally adjusted by a recent level correction.

    Args:
        context (np.array): A 1D numpy array of the PAST target values for one series.
        prediction_length (int): The forecast horizon.
        freq (str): A frequency string, e.g., "H".
        metadata (dict, optional): An optional dictionary with series metadata.

    Returns:
        np.array: A 1D numpy array of length `prediction_length` with the point forecasts.
    """
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    # If no historical data, return zeros as a sane fallback.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: Hourly data has strong daily seasonality (period 24).
    season_period = 24

    # Fallback to last-value naive if not enough data for a full season.
    # This covers cases where n < season_period.
    if n < season_period:
        return np.full(prediction_length, float(context[-1]))

    # Use the last K full seasons (days) to estimate the seasonal pattern.
    # K: number of full seasons to average.
    #   - Must be at least 1.
    #   - Limited to 8 to avoid incorporating very old data that might not be relevant.
    #   - Guaranteed that K * season_period <= n.
    K = min(max(1, n // season_period), 8)

    # Extract the last K * season_period points from the context.
    # These points will be reshaped into a K x season_period matrix.
    # Indexing Safety: context[-K * season_period:] is safe because K * season_period <= n.
    seasonal_data_block = context[-K * season_period:]

    # Calculate the mean of each 'hour' across the K 'days' to get the smoothed seasonal pattern.
    # This also forms the 'base level' of the seasonal component averaged over these K days.
    # Reshape `seasonal_data_block` into `K` rows (representing `K` days) and `season_period` columns (hours).
    # Then, calculate the mean for each column (hour).
    seasonal_pattern = seasonal_data_block.reshape(K, season_period).mean(axis=0)

    # Calculate a level correction to align the forecast with the overall level of the most recent data.
    # Compare the mean of the most recent full season with the mean of the entire `seasonal_data_block`.
    recent_level = context[-season_period:].mean() # Mean of the last full season
    base_seasonal_level = seasonal_data_block.mean() # Mean of the K seasons used for pattern

    # Only apply level correction if both means are valid numbers (not NaN).
    if np.isnan(recent_level) or np.isnan(base_seasonal_level):
        level_correction = 0.0  # No correction if means are NaN
    else:
        level_correction = recent_level - base_seasonal_level

    # Tile the seasonal pattern to cover the prediction length.
    # `reps` is the number of times the `seasonal_pattern` needs to be repeated.
    reps = int(np.ceil(prediction_length / season_period))
    forecast_values = np.tile(seasonal_pattern, reps)[:prediction_length]

    # Apply the calculated level correction to all forecast values.
    forecast_values += level_correction

    # Hard requirement 4: Never output NaN or inf. Clip/guard as needed.
    # Use the last valid context value as a robust fallback for any NaN/inf in the forecast.
    # This ensures a numeric output.
    final_forecast = np.nan_to_num(
        forecast_values,
        nan=float(context[-1]),
        posinf=float(context[-1]),
        neginf=float(context[-1])
    )

    # Ensure the final output type is float.
    return final_forecast.astype(float)