import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Determine a robust fallback value.
    # This value is used for:
    # 1. Forecasts when context is too short for seasonality.
    # 2. Replacing any NaNs/Infs that might arise from calculations.
    fallback_value = 0.0
    if n > 0:
        # Find the last valid value in the context.
        valid_context = context[~np.isnan(context)]
        if len(valid_context) > 0:
            fallback_value = valid_context[-1]
        # If all context values are NaN, fallback_value remains 0.0

    if n == 0:
        # If no context is provided, return an array of fallback_value
        return np.full(prediction_length, fallback_value, dtype=float)

    # Daily seasonality for hourly data is 24. This is a critical benchmark fact.
    season = 24

    if n < season:
        # If there isn't enough data for at least one full season,
        # fall back to a simple last-value naive forecast using the robust fallback_value.
        return np.full(prediction_length, fallback_value, dtype=float)

    # Main seasonal logic: average the last K full days (seasons)
    # K: number of full seasons (days) to average.
    # It's capped at 8 to prevent using overly old data and keep computation lightweight.
    # max(1, ...) ensures K is at least 1 if n >= season.
    K = min(max(1, n // season), 8)

    # Extract the last K full seasons from the context.
    # context[-K * season:] ensures we only use data points corresponding to full seasons
    # ending at the last available data point.
    history_for_seasonal = context[-K * season:]

    # Reshape the history into a K x season matrix.
    # Each row represents a full day, and columns represent hours of the day.
    mat = history_for_seasonal.reshape(K, season)

    # Compute the average seasonal pattern for each hour of the day.
    # np.nanmean is used to gracefully handle any potential NaNs within the historical data
    # by ignoring them in the average calculation.
    seasonal_pattern = np.nanmean(mat, axis=0)

    # If, for some reason, the entire seasonal_pattern becomes NaN
    # (e.g., all values for an hour across all K days were NaN),
    # fall back to a flat forecast of the robust fallback_value.
    if np.all(np.isnan(seasonal_pattern)):
        return np.full(prediction_length, fallback_value, dtype=float)

    # Replicate the learned seasonal pattern to cover the entire prediction_length.
    # Calculate how many times the seasonal pattern needs to be repeated.
    reps = int(np.ceil(prediction_length / season))
    out = np.tile(seasonal_pattern, reps)[:prediction_length]

    # Final NaN/inf guard:
    # Replace any remaining NaNs or Infs in the output forecast with the robust fallback_value.
    # This handles cases where np.nanmean might have produced NaN for a specific hour
    # if all K historical values for that hour were NaN.
    out = np.nan_to_num(out, nan=fallback_value, posinf=fallback_value, neginf=fallback_value)

    return out.astype(float)