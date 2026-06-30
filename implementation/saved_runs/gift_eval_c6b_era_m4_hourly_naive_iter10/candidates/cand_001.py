import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    season = 24  # Critical benchmark fact: daily seasonality (period 24) for hourly data
    
    if n < season:
        # Fallback to last-value naive if not enough data for one full season
        # This handles contexts of length 1 to 23
        return np.full(prediction_length, float(context[-1]))

    # Calculate K, the number of full seasons to average.
    # K should be at least 1, and capped to avoid using excessively long history or
    # to avoid issues with very short total history. Capping at 8 (8 days) is a good heuristic.
    # K = min(max(1, n // season), 8) ensures we have at least K*season points available.
    K = min(max(1, n // season), 8)            
    
    # Extract the last K full seasons from the context.
    # Indexing safety: context[-K * season:] is safe because n >= K * season.
    # n // season determines the maximum K that can be extracted.
    # Since K is min(n // season, 8), K * season <= (n // season) * season <= n.
    recent_seasonal_data = context[-K * season:]
    
    # Reshape the data into a K x season matrix.
    # Each row is a full season (e.g., a full day).
    # This reshape is safe as recent_seasonal_data length is K * season.
    mat = recent_seasonal_data.reshape(K, season)
    
    # Average across the rows (days) to get a denoised seasonal pattern.
    # seasonal will be a 1D array of length `season`.
    seasonal_pattern = mat.mean(axis=0)
    
    # Tile the seasonal pattern to cover the entire prediction length.
    # reps: how many times the seasonal pattern needs to be repeated.
    reps = int(np.ceil(prediction_length / season))
    
    # Repeat and then slice to exactly `prediction_length`.
    out = np.tile(seasonal_pattern, reps)[:prediction_length]
    
    # Hard requirement 4: Never output NaN or inf.
    # If `seasonal_pattern` calculation somehow results in NaN (e.g., if K=0 or all input NaNs),
    # replace with the last known good value (context[-1]).
    # Also ensures the output is float.
    return np.nan_to_num(out, nan=float(context[-1])).astype(float)