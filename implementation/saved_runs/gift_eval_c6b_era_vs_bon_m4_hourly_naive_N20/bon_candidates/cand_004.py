import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Determine a safe fallback value for NaNs, mainly if context[-1] is NaN
    # or if seasonal calculations result in NaN (e.g., all historical data is NaN).
    # We try the mean of valid context values. If no valid values, default to 0.0.
    valid_context = context[~np.isnan(context)]
    if len(valid_context) > 0:
        safe_fallback_value = np.mean(valid_context)
    else:
        safe_fallback_value = 0.0

    # Hard requirement 8: Handle short context arrays robustly.
    # If context is empty, return an array of the safe fallback value.
    if n == 0:
        return np.full(prediction_length, safe_fallback_value, dtype=float)

    # Get the last observed value, guarding against it being NaN itself.
    # This `last_val` is used for fallbacks when context is too short for seasonality,
    # and as a reference for level shifting.
    last_val = context[-1]
    if np.isnan(last_val):
        last_val = safe_fallback_value

    # Critical benchmark: Hourly data has strong daily seasonality (period 24).
    season = 24

    # Fallback for context shorter than one full season.
    # In this case, we cannot compute a seasonal pattern, so we use a last-value naive forecast.
    if n < season:
        return np.full(prediction_length, last_val, dtype=float)

    # Calculate K: the number of past full seasons to average.
    # We use at least 1 full season (since n >= season) and at most 8 to prevent
    # averaging too much distant or noisy history.
    K = min(max(1, n // season), 8)

    # Extract the last K *full* seasons from the context.
    # Hard requirement 9: Indexing safety. context[-K * season:] is safe because K * season <= n.
    data_for_season_avg = context[-K * season:]
    
    # Reshape the data into K rows (representing each season) and 'season' columns (each hour of the day).
    mat = data_for_season_avg.reshape(K, season)
    
    # Calculate the average seasonal pattern.
    # Use np.nanmean to safely compute the mean, ignoring any NaNs in the historical data.
    seasonal_pattern = np.nanmean(mat, axis=0)

    # If, for some reason, all values for a specific hour in `mat` were NaN, `np.nanmean`
    # would return NaN for that position. We fill these remaining NaNs with the safe fallback value.
    seasonal_pattern = np.nan_to_num(seasonal_pattern, nan=safe_fallback_value)

    # Calculate a level shift to adapt the seasonal forecast to the most recent observation.
    # This helps in models where the general level of the series might have shifted.
    
    # The last observed value belongs to the current (possibly incomplete) seasonal cycle.
    # Its position within that cycle is (n - 1) % season.
    last_seasonal_idx = (n - 1) % season
    
    # Get the average value for this specific seasonal position from our calculated pattern.
    avg_at_last_idx = seasonal_pattern[last_seasonal_idx]
    
    # The level shift is the difference between the actual last observed value and
    # the historical average for that specific seasonal point.
    level_shift = last_val - avg_at_last_idx

    # Tile the calculated seasonal pattern to cover the entire prediction_length horizon.
    # Repeat the pattern enough times and then trim to the exact prediction_length.
    reps = int(np.ceil(prediction_length / season))
    tiled_seasonal_forecast = np.tile(seasonal_pattern, reps)[:prediction_length]

    # Apply the calculated level shift to the tiled seasonal forecast.
    point_forecasts = tiled_seasonal_forecast + level_shift
    
    # Hard requirement 4: Never output NaN or inf.
    # Use np.nan_to_num one last time to catch any potential NaNs or Infs that might
    # have been introduced (e.g., if level_shift somehow became inf or nan).
    # Replace them with our robust safe_fallback_value.
    point_forecasts = np.nan_to_num(point_forecasts, nan=safe_fallback_value).astype(float)
    
    # Hard requirement 3: Output length must be EXACTLY prediction_length.
    # This is ensured by the tiling and slicing above.
    return point_forecasts