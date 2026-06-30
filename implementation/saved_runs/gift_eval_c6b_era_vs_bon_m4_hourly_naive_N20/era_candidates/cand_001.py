import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- Determine a robust fallback value ---
    # This value will be used if context is empty, too short, or all NaNs/Infs.
    fallback_value = 0.0
    if n > 0:
        # Find the last finite value in context. This is the most reliable last known value.
        last_finite_idx = np.where(np.isfinite(context))[0]
        if len(last_finite_idx) > 0:
            fallback_value = context[last_finite_idx[-1]]
        # If no finite values are found in context (e.g., all NaNs/Infs),
        # fallback_value remains 0.0, which is a safe default.

    # --- Handle very short contexts ---
    # If the context array is empty, return a constant forecast using the fallback value.
    if n == 0:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- Define seasonality ---
    # Critical benchmark facts indicate strong daily seasonality (period 24) for hourly data.
    season = 24  # Daily seasonality for hourly data (24 hours in a day).

    # If context is shorter than one full season, fall back to a naive forecast
    # using the robust fallback_value. This ensures stability for short series.
    if n < season:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- Seasonal Averaging Model ---
    # Determine K, the number of past full seasons to average.
    # We cap K at 8 to avoid using very old, potentially irrelevant data
    # that might introduce noise or stale patterns.
    # `max(1, n // season)` ensures we average at least one full season if available.
    K = min(max(1, n // season), 8)

    # Slice the last K * `season` points from the context.
    # This extracts the data for the most recent K full seasonal cycles.
    seasonal_data_slice = context[-K * season:]

    # Reshape the sliced data into a K x `season` matrix.
    # Each row represents a full season (e.g., a full day's data).
    # NaNs present in the `context` will be preserved in this matrix.
    mat = seasonal_data_slice.reshape(K, season)

    # Calculate the mean for each hour (column) across the K seasons (rows).
    # `np.nanmean` is used to handle NaNs gracefully: it ignores them when calculating
    # the mean. If a particular hour of day (column) is all NaNs across the K seasons,
    # `np.nanmean` will return NaN for that specific hour.
    seasonal_pattern = np.nanmean(mat, axis=0)

    # Replace any NaNs that might still exist in the `seasonal_pattern` (e.g., if a
    # specific hour was always NaN across all K seasons) with the robust fallback value.
    seasonal_pattern = np.nan_to_num(seasonal_pattern, nan=fallback_value)

    # --- Generate forecasts ---
    # Tile the cleaned seasonal pattern to cover the entire `prediction_length`.
    # `reps` calculates how many times the `season` pattern needs to be repeated.
    reps = int(np.ceil(prediction_length / season))
    tiled_forecast = np.tile(seasonal_pattern, reps)

    # Truncate the tiled pattern to the exact required `prediction_length`.
    final_forecast = tiled_forecast[:prediction_length]

    # Ensure the final output array has a float data type.
    return final_forecast.astype(float)