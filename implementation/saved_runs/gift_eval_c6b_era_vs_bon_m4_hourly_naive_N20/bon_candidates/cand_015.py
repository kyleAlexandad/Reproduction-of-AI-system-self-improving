import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle extremely short contexts, including empty (length 0).
    if n == 0:
        # If no history, return zeros as a robust fallback.
        return np.zeros(prediction_length, dtype=float)

    # Define the primary seasonal period for hourly data, which is daily (24 hours).
    season = 24

    # If the context is shorter than one full season, a seasonal forecast is not possible.
    # Fall back to a simple last-value naive forecast.
    if n < season:
        return np.full(prediction_length, float(context[-1]))

    # Determine K, the number of full past seasons to average.
    # We use at least 1 season, and cap it at 8 seasons to balance robustness and recency.
    # K is also limited by the actual number of full seasons available in the context.
    K = min(max(1, n // season), 8)

    # Extract the last K full seasons from the context data.
    # The slice `context[-K * season:]` is safe because `K * season` is guaranteed
    # to be less than or equal to `n` (as K <= n // season).
    seasonal_data_matrix = context[-K * season:].reshape(K, season)

    # Calculate the average pattern for each hour of the day across the K seasons.
    # This results in a 1D array of length `season` (e.g., 24 for daily pattern).
    seasonal_pattern = seasonal_data_matrix.mean(axis=0)

    # Generate the raw forecast by tiling the averaged seasonal pattern.
    # We tile enough repetitions to cover the entire `prediction_length`.
    reps = int(np.ceil(prediction_length / season))
    tiled_pattern = np.tile(seasonal_pattern, reps)

    # Slice the tiled pattern to ensure the output has exactly `prediction_length` values.
    forecast_output = tiled_pattern[:prediction_length]

    # Robustness check: ensure no NaN or inf values are returned.
    # `np.nan_to_num` replaces NaNs with 0 (or a specified value) and inf with large finite numbers.
    # Using `float(context[-1])` as the `nan` fallback provides a sensible default if any
    # part of the seasonal calculation somehow resulted in a NaN (e.g., from an all-NaN column in matrix).
    forecast_output = np.nan_to_num(forecast_output, nan=float(context[-1])).astype(float)

    return forecast_output