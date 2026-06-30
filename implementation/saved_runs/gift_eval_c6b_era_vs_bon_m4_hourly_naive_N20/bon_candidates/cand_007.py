import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    if n == 0:
        # Handle empty context: return zeros.
        return np.zeros(prediction_length, dtype=float)

    # Daily seasonality is crucial for hourly data, period 24 hours.
    season = 24 

    if n < season:
        # If context is shorter than one full season, fall back to last-value naive forecast.
        # This is a robust fallback for very short series.
        return np.full(prediction_length, float(context[-1]))

    # Determine 'K': the number of past full seasons (days) to average for the seasonal pattern.
    # We aim to average between 1 and 8 full past days to denoise the seasonal pattern.
    # 'n // season' gives the number of available full seasons.
    # min(..., 8) caps it to avoid using excessively long history if 'n' is very large.
    # max(1, ...) ensures we always use at least one full season if 'n >= season'.
    K = min(max(1, n // season), 8)

    # Extract the last K * season data points.
    # Reshape them into a K x season matrix, where each row represents a full past day.
    # Example: if K=3 and season=24, this takes the last 72 hours and reshapes into 3 rows of 24 hours.
    mat = context[-K * season:].reshape(K, season)

    # Compute the average value for each hour of the day across the K past days.
    # This gives us a denoised seasonal pattern of length 'season'.
    seasonal_pattern = mat.mean(axis=0)

    # Tile this seasonal pattern to cover the entire prediction_length.
    # Calculate how many repetitions of the seasonal_pattern are needed.
    reps = int(np.ceil(prediction_length / season))
    
    # Tile the pattern and then trim it to the exact prediction_length.
    forecast_output = np.tile(seasonal_pattern, reps)[:prediction_length]

    # Robustness: ensure no NaN or infinite values are present in the output.
    # If, for some reason, `seasonal_pattern` (or `forecast_output`) contained NaN,
    # replace them with the last observed value from the context.
    forecast_output = np.nan_to_num(forecast_output, nan=float(context[-1]))

    return forecast_output.astype(float)