import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Hourly data has strong daily seasonality (period 24)
    season = 24 

    # Fallback to last-value naive if context is too short for a full season
    if n < season:
        # np.full is safer than np.repeat(context[-1], ...) for potential edge cases with context[-1]
        return np.full(prediction_length, float(context[-1]))

    # Determine K, the number of full seasons to average.
    # Use at least 1 full season, up to a maximum of 8 seasons,
    # constrained by available data (n // season).
    K = min(max(1, n // season), 8)

    # Extract the last K full seasons from the context.
    # We slice `context[-K * season:]` to get the most recent K * season points.
    # If n < K * season, n // season would be smaller, so K is adjusted.
    # For instance, if n=25 and season=24, K=1, so context[-24:] is used.
    recent_seasonal_data = context[-K * season:]

    # Reshape the data into (K, season) to align the seasonal cycles.
    # Each row represents a full season.
    mat = recent_seasonal_data.reshape(K, season)

    # Calculate the average pattern for each hour of the day across K seasons.
    seasonal_pattern = mat.mean(axis=0)

    # Tile the averaged seasonal pattern to cover the entire prediction_length.
    # Calculate how many repetitions are needed.
    reps = int(np.ceil(prediction_length / season))
    
    # Generate the raw forecast by tiling and then truncate to prediction_length.
    out = np.tile(seasonal_pattern, reps)[:prediction_length]

    # Ensure no NaN or Inf values are returned. Replace NaNs with the last known value from context.
    # The mean operation on floats should not produce NaNs unless the input itself contains NaNs.
    # If context has NaNs, context[-1] might be NaN. A more robust fallback could be a mean of last few values.
    # However, for this task, context is expected to be valid numerical data.
    # np.nan_to_num handles both NaN and inf.
    return np.nan_to_num(out, nan=float(context[-1]), posinf=float(context[-1]), neginf=float(context[-1])).astype(float)