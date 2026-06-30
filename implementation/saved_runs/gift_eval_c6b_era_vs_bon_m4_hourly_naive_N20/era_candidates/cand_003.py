import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- 1. Determine a robust fallback value ---
    # This value will be used if context is empty, too short, all NaNs/Infs,
    # or to fill NaNs in the derived seasonal pattern.
    fallback_value = 0.0
    if n > 0:
        # Find the last finite value in context. This is the most reliable last known value.
        last_finite_idx = np.where(np.isfinite(context))[0]
        if len(last_finite_idx) > 0:
            fallback_value = context[last_finite_idx[-1]]
    # Ensure fallback_value itself is finite, defaulting to 0.0 if not.
    if not np.isfinite(fallback_value):
        fallback_value = 0.0

    # --- 2. Handle very short contexts ---
    # If the context array is empty, return a constant forecast using the fallback value.
    if n == 0:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 3. Define seasonality ---
    # Critical benchmark facts indicate strong daily seasonality (period 24) for hourly data.
    season = 24  # Daily seasonality for hourly data (24 hours in a day).

    # If context is shorter than one full season, fall back to a naive forecast
    # using the robust fallback_value. This ensures stability and is effectively last-value naive
    # for such short series as there's no full seasonal pattern to extract.
    if n < season:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 4. Derive the seasonal pattern by averaging the last K full seasons ---
    # This approach is generally more robust to noise and individual outliers than
    # relying solely on the last season, as suggested by the prompt ("average the last K full days").

    # Determine K, the number of full seasons to average.
    # Cap K at 8 to avoid using excessively old data, and ensure at least 1 season is used
    # if available (handled by `n < season` check already).
    # `n // season` gives the total number of full seasons available in the context.
    K = min(max(1, n // season), 8)

    # Extract the last K full seasons from the context.
    # This slice will have K * season points.
    history_for_seasonal_avg = context[-K * season:]

    # Reshape the history into K rows (each row is a season) and `season` columns (each column is an hour).
    mat = history_for_seasonal_avg.reshape(K, season)

    # Calculate the seasonal pattern by averaging values for each hour across K seasons.
    # np.nanmean handles NaNs by ignoring them during the average calculation.
    seasonal_pattern = np.nanmean(mat, axis=0)

    # --- Robustness check for seasonal_pattern ---
    # If a specific hour (column) was entirely NaN across all K seasons, nanmean will return NaN.
    # We need to fill these remaining NaNs with the robust fallback_value.
    seasonal_pattern[np.isnan(seasonal_pattern)] = fallback_value

    # Final safety net to ensure all values in the seasonal pattern are finite.
    seasonal_pattern = np.nan_to_num(seasonal_pattern, nan=fallback_value)

    # --- 5. Generate forecasts ---
    # Tile the cleaned and robust seasonal pattern to cover the entire `prediction_length`.
    reps = int(np.ceil(prediction_length / season))
    tiled_forecast = np.tile(seasonal_pattern, reps)

    # Truncate the tiled pattern to the exact required `prediction_length`.
    final_forecast = tiled_forecast[:prediction_length]

    # Ensure the final output array has a float data type.
    return final_forecast.astype(float)