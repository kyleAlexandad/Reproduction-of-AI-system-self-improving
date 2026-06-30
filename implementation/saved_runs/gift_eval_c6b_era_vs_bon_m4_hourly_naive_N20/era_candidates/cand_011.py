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

    # --- 3. Extract the Last Full Daily Season for Seasonal Naive Forecast ---
    # The core of this strategy is to directly repeat the last complete daily season observed
    # in the context. This aligns with the "SEASONAL-NAIVE forecast with DAILY period 24 is STRONG"
    # benchmark fact.
    # The slice `context[-season_daily:]` is guaranteed to be valid because `n >= season_daily`
    # has already been checked.
    last_full_season_data = context[-season_daily:]
    
    # --- 4. Robustify the Seasonal Pattern ---
    # It's possible that values within the `last_full_season_data` array are NaN or Inf.
    # These non-finite values must be replaced to ensure a valid and stable forecast.
    # The `fallback_value` is used for imputation, providing a sane default for any missing hours
    # within the last observed season.
    seasonal_pattern = np.nan_to_num(last_full_season_data, nan=fallback_value)

    # --- 5. Generate Forecasts ---
    # Tile the robust 24-point seasonal pattern to cover the entire `prediction_length`.
    # `reps` determines how many full repetitions of the seasonal pattern are needed.
    reps = int(np.ceil(prediction_length / season_daily))
    
    # Tile the pattern and then slice to exactly `prediction_length` values.
    forecast_output = np.tile(seasonal_pattern, reps)[:prediction_length]

    # --- 6. Final Robustness Check ---
    # As a final safeguard, ensure that no NaN or Inf values have inadvertently
    # appeared in the `forecast_output` (e.g., from unexpected floating-point edge cases).
    # Replace any such values with the `fallback_value` to ensure a fully valid output.
    final_forecast = np.nan_to_num(forecast_output, nan=fallback_value)
    
    # Return the forecast as a float array of the specified prediction_length.
    return final_forecast.astype(float)
