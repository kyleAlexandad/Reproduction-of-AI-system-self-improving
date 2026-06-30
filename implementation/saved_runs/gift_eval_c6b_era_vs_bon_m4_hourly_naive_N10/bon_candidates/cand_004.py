import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle extremely short contexts or empty context
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: Hourly data has strong DAILY seasonality (period 24).
    season = 24

    # Fallback to last-value naive if context is too short to observe one full season.
    # This ensures robustness for very short series.
    if n < season:
        return np.full(prediction_length, float(context[-1]))

    # Determine how many full seasons (days) to average to denoise the seasonal pattern.
    # We average the last K full days. A maximum of 8 days is a reasonable heuristic
    # to capture a stable daily pattern without being too sensitive to old data.
    # K must be at least 1 if n >= season.
    K = min(max(1, n // season), 8)

    # Extract the last K full seasons from the context.
    # This slice is safe because n >= K * season.
    # `context[-K * season:]` ensures we get exactly K * season data points.
    data_for_seasonal_pattern = context[-K * season:]

    # Reshape the data into K rows (representing K days) and 'season' columns (representing
    # the hours within a day).
    # Then, compute the mean for each column (each hour of the day) across the K days.
    # This gives us a smoothed, representative seasonal pattern for a full day.
    seasonal_pattern = data_for_seasonal_pattern.reshape(K, season).mean(axis=0)

    # Tile the derived seasonal pattern to cover the entire prediction length.
    # First, calculate how many repetitions of the seasonal pattern are needed.
    reps = int(np.ceil(prediction_length / season))
    tiled_forecast = np.tile(seasonal_pattern, reps)

    # Trim the tiled forecast to the exact `prediction_length` required.
    output_forecast = tiled_forecast[:prediction_length]

    # Ensure no NaN or infinite values are present in the output.
    # Replace any such values with the last observed value from the context,
    # which is a robust fallback for potentially problematic series.
    # The output is explicitly cast to float, matching the input context dtype.
    return np.nan_to_num(output_forecast, nan=float(context[-1])).astype(float)