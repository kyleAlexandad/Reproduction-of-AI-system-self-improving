import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- 1. Determine a robust fallback value ---
    # This value is crucial for handling empty, very short, or entirely NaN/Inf contexts.
    # It attempts to use the last known finite value from the context, otherwise defaults to 0.0.
    fallback_value = 0.0
    if n > 0:
        # Find indices of all finite values in the context.
        last_finite_idx = np.where(np.isfinite(context))[0]
        if len(last_finite_idx) > 0:
            # If finite values exist, the last one is the most reliable recent value.
            fallback_value = context[last_finite_idx[-1]]
    # Ensure fallback_value itself is finite, handling cases where context might contain only NaNs/Infs.
    if not np.isfinite(fallback_value):
        fallback_value = 0.0

    # Define the primary daily seasonality for hourly data.
    # The critical benchmark facts strongly indicate period 24 as the dominant and strong seasonality.
    season_daily = 24

    # --- 2. Handle very short contexts ---
    # If the context is empty or shorter than a full daily season,
    # we cannot establish a reliable seasonal pattern. In these cases, return a constant forecast
    # using the robust fallback value. This effectively acts as a last-value naive forecast
    # for scenarios with insufficient historical data.
    if n == 0 or n < season_daily:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 3. Determine the number of past seasons (K) to average ---
    # The prompt recommends averaging the last K full days to denoise the seasonal pattern.
    # K is chosen to be at least 1 (if a full season is available), and capped at 8
    # to avoid using too much potentially stale historical data.
    K = min(max(1, n // season_daily), 8)

    # --- 4. Extract and average the last K full daily seasons ---
    # We slice the context to get the most recent `K * season_daily` points.
    # This slice is guaranteed to be valid because n >= season_daily and K <= n // season_daily.
    history_for_seasonal_average = context[-K * season_daily:]

    # Reshape the data into a (K, season_daily) matrix, where each row is a full season.
    # Then, calculate the mean across rows (axis=0) to get the averaged hourly pattern.
    seasonal_matrix = history_for_seasonal_average.reshape(K, season_daily)
    averaged_seasonal_pattern = np.mean(seasonal_matrix, axis=0)

    # --- 5. Robustify the Seasonal Pattern ---
    # It's possible that values within the `averaged_seasonal_pattern` array are NaN or Inf
    # if, for example, a specific hour across all `K` seasons was NaN.
    # These non-finite values must be replaced to ensure a valid and stable forecast.
    # The `fallback_value` is used for imputation.
    seasonal_pattern_robust = np.nan_to_num(averaged_seasonal_pattern, nan=fallback_value)

    # --- 6. Generate Forecasts ---
    # Tile the robust 24-point seasonal pattern to cover the entire `prediction_length`.
    # `reps` determines how many full repetitions of the seasonal pattern are needed.
    reps = int(np.ceil(prediction_length / season_daily))

    # Tile the pattern and then slice to exactly `prediction_length` values.
    forecast_output = np.tile(seasonal_pattern_robust, reps)[:prediction_length]

    # --- 7. Final Robustness Check ---
    # As a final safeguard, ensure that no NaN or Inf values have inadvertently
    # appeared in the `forecast_output`. Replace any such values with the `fallback_value`
    # to ensure a fully valid output.
    final_forecast = np.nan_to_num(forecast_output, nan=fallback_value)

    # Return the forecast as a float array of the specified prediction_length.
    return final_forecast.astype(float)