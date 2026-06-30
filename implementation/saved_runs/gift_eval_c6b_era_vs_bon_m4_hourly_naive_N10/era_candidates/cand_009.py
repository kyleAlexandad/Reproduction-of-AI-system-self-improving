import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    if n == 0:
        # If no history, return zeros as a safe default.
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: Hourly data has strong daily seasonality (period 24).
    season_daily = 24 # Daily seasonality for hourly data

    # --- Fallbacks and Core Seasonal Forecast ---
    if n < season_daily:
        # Fallback 1: Last-value naive if context is shorter than a full daily season.
        # This handles cases where seasonality cannot be extracted.
        # Hard requirement 9: Indexing safety. context[-1] is safe as n > 0.
        forecast_values = np.full(prediction_length, float(context[-1]), dtype=float)
    else:
        # Core forecast: Simple Seasonal Naive using the last full daily season.
        # The prompt states that "A SEASONAL-NAIVE forecast with DAILY period 24 is STRONG: MASE ~= 1.19".
        # This is significantly better than other simple baselines like last-value naive or damped trend.
        #
        # The parent candidate attempted to add a "Damped Level Correction" on top of this
        # simple seasonal naive approach, but its MASE (1.223402) was worse than the
        # reported MASE for simple seasonal naive (~1.19).
        # The prompt also warned that "damped trend (~12.04) are NOT better than naive."
        # This strongly suggests that for this specific dataset, adding any form of
        # trend or level correction *degrades* performance compared to a pure seasonal naive.
        #
        # Therefore, this improved version focuses on directly implementing the strong
        # simple seasonal naive (period 24) baseline, without additional modifications
        # that have proven to be detrimental for this benchmark.

        # Hard requirement 9: Indexing safety. context[-season_daily:] is safe because n >= season_daily.
        # Extract the last full daily season directly.
        seasonal_pattern = context[-season_daily:]
        
        # Tile the seasonal pattern to cover the entire prediction_length.
        num_repetitions = int(np.ceil(prediction_length / season_daily))
        forecast_values = np.tile(seasonal_pattern, num_repetitions)[:prediction_length]

        # Removed the "Damped Level Correction" and K-averaged seasonal pattern logic
        # from the parent and suggested template, respectively, as they did not
        # outperform the simple seasonal naive for this benchmark.

    # Hard requirement 4: Never output NaN or inf.
    # np.nan_to_num replaces any NaNs (e.g., if context contained NaNs that propagated through mean calculations)
    # with a safe fallback value (the last observed value from context).
    # Finally, ensure the output type is float.
    # Hard requirement 9: Indexing safety for context[-1] when n >= season_daily (or n > 0 in fallback).
    return np.nan_to_num(forecast_values, nan=float(context[-1])).astype(float)