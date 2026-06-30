import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle extremely short contexts
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Daily seasonality for hourly data
    season = 24

    # If context is shorter than one full season, fall back to last-value naive
    if n < season:
        return np.full(prediction_length, float(context[-1]))

    # Determine K: number of past full seasons to average.
    # We average a minimum of 1 season, up to a maximum of 8 seasons,
    # constrained by the available history.
    K = min(max(1, n // season), 8)

    # Extract the last K full seasons from the context.
    # The slice context[-K * season:] is safe because K * season <= n.
    # Reshape it into a K rows x 'season' columns matrix.
    # Each row represents a full day's pattern.
    mat = context[-K * season:].reshape(K, season)

    # Calculate the average pattern across the K seasons.
    # This denoised pattern captures the typical daily cycle.
    seasonal_pattern = mat.mean(axis=0)

    # Tile the seasonal pattern to cover the entire prediction length.
    reps = int(np.ceil(prediction_length / season))
    out = np.tile(seasonal_pattern, reps)[:prediction_length]

    # Ensure no NaN or Inf values are returned. Replace NaNs with the last context value.
    # Although `mat.mean(axis=0)` should not produce NaN unless input itself is NaN,
    # this is a robust guard.
    return np.nan_to_num(out, nan=float(context[-1])).astype(float)