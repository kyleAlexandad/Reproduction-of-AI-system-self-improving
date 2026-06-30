import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    season = 24  # Critical benchmark fact: hourly data has strong DAILY seasonality (period 24)
    
    # Fallback to last-value naive if context is shorter than one full season
    # This also handles cases where n=1 to n=23.
    if n < season:
        return np.full(prediction_length, float(context[-1]))
    
    # Determine K, the number of past full seasons to average.
    # K is at least 1 (if n >= season) and capped at 8 to prevent using overly stale history.
    # This strategy helps denoise the seasonal pattern.
    K = min(max(1, n // season), 8)
    
    # Extract the last K * full* seasons from the context.
    # Indexing is safe: n >= K * season is guaranteed by the calculation of K.
    recent_seasonal_data = context[-K * season:]
    
    # Reshape the data into a (K, season) matrix, where each row is a full season.
    mat = recent_seasonal_data.reshape(K, season)
    
    # Calculate the average seasonal pattern by averaging across the K seasons (rows).
    seasonal_pattern = mat.mean(axis=0)
    
    # --- Level Adjustment Enhancement ---
    # Adjust the level of the forecast based on the most recent full season's average.
    # This makes the forecast responsive to recent shifts in the overall level.
    
    # Calculate the average level of the most recent *full* season available in context.
    # This is safe because n >= season is guaranteed by the earlier check.
    last_season_avg = np.mean(context[-season:])
    
    # Calculate the average level of the derived seasonal pattern.
    seasonal_pattern_avg = np.mean(seasonal_pattern)
    
    # The level shift is the difference between the most recent average and the pattern's average.
    level_shift = last_season_avg - seasonal_pattern_avg
    
    # Apply the calculated level shift to the seasonal pattern.
    shifted_seasonal_pattern = seasonal_pattern + level_shift
    
    # --- Generate Forecast ---
    # Tile the shifted seasonal pattern to cover the entire prediction_length.
    # `reps` determines how many times the `season` pattern needs to be repeated.
    reps = int(np.ceil(prediction_length / season))
    out = np.tile(shifted_seasonal_pattern, reps)[:prediction_length]
    
    # Hard requirement 3: Output length must be EXACTLY prediction_length.
    # Hard requirement 4: Never output NaN or inf. Clip/guard as needed.
    # Use nan_to_num to replace any potential NaNs with the last observed context value,
    # which is a robust fallback for unexpected data issues. Ensure float type.
    return np.nan_to_num(out, nan=float(context[-1])).astype(float)
