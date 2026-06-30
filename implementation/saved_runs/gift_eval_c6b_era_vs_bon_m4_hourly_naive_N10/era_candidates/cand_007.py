import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Determine a safe fallback value: the last non-NaN observation.
    # This is used for very short contexts and as a fill value for np.nan_to_num.
    fallback_value = 0.0 # Default if no valid observations at all
    valid_context = context[~np.isnan(context)]
    if len(valid_context) > 0:
        fallback_value = valid_context[-1]

    # Critical benchmark fact: Hourly data has strong daily seasonality (period 24).
    season = 24

    if n < season:
        # Fallback 1: Last-valid-value naive if context is shorter than a full daily season.
        # This handles cases where seasonality cannot be extracted.
        return np.full(prediction_length, fallback_value, dtype=float)

    # Core seasonal forecast: Average the last K full days to denoise the pattern.
    # K: number of full seasons to average. Max 8 days (for 8 weeks) to balance smoothing
    # with retaining recent patterns. K is at least 1 because n >= season here.
    K = min(max(1, n // season), 8)

    # Extract the last K full seasons from the context.
    # Indexing safety: K * season is guaranteed to be <= n here.
    seasonal_data_slice = context[-K * season:]

    # Reshape into K rows (days) and `season` columns (hours of the day).
    mat = seasonal_data_slice.reshape(K, season)

    # Compute the mean for each hour of the day, ignoring NaNs (robustness).
    # If all values for a particular hour are NaN, seasonal_pattern will have NaN at that position.
    seasonal_pattern = np.nanmean(mat, axis=0)

    # Tile the averaged seasonal pattern to cover the entire prediction_length.
    num_repetitions = int(np.ceil(prediction_length / season))
    forecast_values = np.tile(seasonal_pattern, num_repetitions)[:prediction_length]

    # Hard requirement 4: Never output NaN or inf.
    # Replace any remaining NaNs (e.g., if a whole hour column was NaN) with the fallback value.
    # Ensure the output array has a float dtype.
    return np.nan_to_num(forecast_values, nan=fallback_value).astype(float)