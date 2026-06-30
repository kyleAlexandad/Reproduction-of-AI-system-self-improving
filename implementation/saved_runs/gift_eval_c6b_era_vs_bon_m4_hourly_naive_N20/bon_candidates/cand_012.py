import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle SHORT context arrays robustly
    if n == 0:
        # If no context, return an array of zeros.
        return np.zeros(prediction_length, dtype=float)

    # Critical Benchmark Fact: Hourly data has strong daily seasonality (period 24).
    season = 24

    # If context is shorter than one full season, fall back to last-value naive forecast.
    # This also ensures indexing safety for seasonal calculations.
    if n < season:
        return np.full(prediction_length, float(context[-1]))

    # Determine K: the number of past full seasons (days) to average.
    # We use a maximum of 8 full seasons to avoid using very old data,
    # and ensure at least one full season is used since `n >= season`.
    # K will be min(total full seasons available in context, 8).
    K = min(n // season, 8)
    
    # If K somehow becomes 0 (should not happen if n >= season), fallback to last value.
    if K == 0:
        return np.full(prediction_length, float(context[-1]))

    # Extract the last K full seasons from the context.
    # Hard requirement 9: INDEXING SAFETY. Slicing from the end.
    # This slice is safe because K * season will always be <= n.
    seasonal_data = context[-K * season:]
    
    # Reshape the data into K rows (representing K past days) and 'season' columns (hours of the day).
    mat = seasonal_data.reshape(K, season)
    
    # Calculate the average seasonal pattern for each 'hour of the day'.
    # Use np.nanmean to be robust against potential NaN values in the historical data,
    # ensuring the average is computed only from non-NaN values.
    seasonal_pattern = np.nanmean(mat, axis=0)

    # --- Level Adjustment ---
    # Apply a level correction to the seasonal forecast.
    # This adjusts the forecast by the difference between the most recent observed value
    # and what its corresponding seasonal component suggests it should be.
    
    # Find the seasonal component that corresponds to the last observed point (context[-1]).
    # The index within the season is (n - 1) % season.
    last_observed_seasonal_component = seasonal_pattern[(n - 1) % season]
    
    # Calculate the level offset.
    # If `last_observed_seasonal_component` is NaN (e.g., if `seasonal_pattern` has NaNs due to
    # all inputs for that specific hour being NaN), default the offset to 0.
    if np.isnan(last_observed_seasonal_component):
        level_offset = 0.0
    else:
        level_offset = context[-1] - last_observed_seasonal_component

    # Generate the initial forecasts by tiling the average seasonal pattern.
    # `reps` calculates how many times the `season` pattern needs to be repeated
    # to cover at least the `prediction_length`.
    reps = int(np.ceil(prediction_length / season))
    out = np.tile(seasonal_pattern, reps)[:prediction_length]

    # Apply the calculated level offset to all forecast points.
    out += level_offset

    # Hard requirement 4: Never output NaN or inf.
    # Use np.nan_to_num to replace any NaN or inf values in the output array.
    # We use 0.0 as a robust fallback for NaN values.
    # For positive/negative infinity, we clamp to large finite values.
    return np.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10).astype(float)
