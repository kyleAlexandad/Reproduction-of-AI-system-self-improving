import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)
    
    # Handle empty context: return zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # Define the primary daily seasonality for hourly data
    season = 24  
    
    # Fallback for very short context: use last observed value (Naive forecast)
    # This applies if the history is shorter than one full season.
    if n < season:
        return np.full(prediction_length, float(context[-1]))
    
    # Determine the number of past full seasons to average.
    # We want at least 1 season, and at most 8 seasons, to create a robust average pattern.
    K = min(max(1, n // season), 8)
    
    # Extract the relevant historical data: the last K full seasons.
    # Slicing from the end ensures safety and gets the most recent data.
    relevant_context = context[-K * season:]
    
    # Reshape the data into K rows (each representing a full season) and 'season' columns
    # (each column representing a specific hour of the day).
    mat = relevant_context.reshape(K, season)
    
    # Calculate the average pattern for each hour of the day by taking the mean across rows.
    # This smooths out noise by averaging multiple past seasonal cycles.
    seasonal_pattern = mat.mean(axis=0)
    
    # Calculate the mean level of the most recent full season.
    # This is used to adjust the averaged seasonal pattern to the current level of the series.
    last_season_mean = context[-season:].mean()
    
    # Calculate the mean level of the derived (averaged) seasonal pattern.
    average_seasonal_pattern_mean = seasonal_pattern.mean()
    
    # Compute the level shift: the difference between the most recent season's mean
    # and the mean of the averaged seasonal pattern.
    level_shift = last_season_mean - average_seasonal_pattern_mean
    
    # Tile the seasonal pattern to cover the entire prediction horizon.
    # We calculate how many repetitions are needed.
    reps = int(np.ceil(prediction_length / season))
    tiled_pattern = np.tile(seasonal_pattern, reps)
    
    # Trim the tiled pattern to the exact `prediction_length`.
    # Then, apply the calculated level shift to adjust the forecast.
    out = tiled_pattern[:prediction_length] + level_shift
    
    # Robustness: Ensure no NaN or Inf values are returned.
    # If any appear (e.g., due to division by zero if context had NaNs, though unlikely here),
    # replace them with the last observed value from the context.
    # Ensure the output type is float.
    return np.nan_to_num(out, nan=float(context[-1])).astype(float)
