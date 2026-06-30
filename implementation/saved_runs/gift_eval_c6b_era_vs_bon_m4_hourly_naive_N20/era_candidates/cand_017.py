import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- 1. Robust Fallback Value ---
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

    # Define the primary seasonal period for hourly data.
    # The critical benchmark facts strongly indicate daily seasonality (24 hours) as dominant.
    season_daily = 24

    # --- 2. Handle very short contexts ---
    # If the context is empty, return an array of zeros to satisfy the output length requirement.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # If the context is shorter than a full daily season, we cannot establish a reliable seasonal pattern.
    # In these cases, return a constant forecast using the robust fallback value.
    # This effectively acts as a last-value naive forecast (or 0.0 if no valid last value).
    if n < season_daily:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 3. Core Forecasting Logic: Robust Seasonal Naive (Period 24) ---
    # The critical benchmark facts explicitly state:
    # "A SEASONAL-NAIVE forecast with DAILY period 24 is STRONG: MASE ~= 1.19 (about 10x better)."
    # This strategy directly implements that finding: using the last full daily cycle as the forecast pattern.
    # This is often more effective than averaging multiple past seasons if the pattern is dynamic
    # or if recent changes are more indicative.

    # Extract the last full daily season from the context.
    # This slice is guaranteed to be valid because `n >= season_daily` is checked above.
    last_season_pattern = context[-season_daily:]
    
    # Robustify the extracted pattern: replace any NaN or Inf values with the `fallback_value`.
    # This ensures that the generated forecast will not contain any invalid numbers.
    cleaned_pattern = np.nan_to_num(last_season_pattern, nan=fallback_value, posinf=fallback_value, neginf=fallback_value)

    # Tile the cleaned daily pattern to cover the entire prediction length.
    # Calculate how many repetitions are needed to cover `prediction_length`.
    num_repetitions = int(np.ceil(prediction_length / season_daily))
    
    # Create the full forecast by tiling the cleaned pattern.
    # Then, truncate the tiled array to the exact `prediction_length`.
    forecast_output = np.tile(cleaned_pattern, num_repetitions)[:prediction_length]

    # --- 4. Final Robustness Check ---
    # As a final safeguard, ensure that no NaN or Inf values have inadvertently
    # appeared in the `forecast_output`. Although `nan_to_num` was applied to the pattern,
    # this provides an extra layer of safety. Replace them with `fallback_value` if any.
    final_forecast = np.nan_to_num(forecast_output, nan=fallback_value, posinf=fallback_value, neginf=fallback_value)
    
    # Return the final forecast as a float array of the specified prediction_length.
    return final_forecast.astype(float)