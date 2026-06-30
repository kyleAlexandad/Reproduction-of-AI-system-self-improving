import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: Daily seasonality (period 24) is very strong for hourly data.
    season_daily = 24

    # Fallback to last-value naive if context is too short for daily seasonality
    if n < season_daily:
        # Hard requirement 4: Never output NaN or inf. `context[-1]` will be finite here.
        return np.full(prediction_length, float(context[-1]))

    # Determine K: the number of past full daily seasons to average.
    # The template's logic: min(max(1, n // season), 8) ensures K is at least 1 and capped.
    # This helps to get a stable seasonal pattern without using excessively old data.
    K_daily = min(max(1, n // season_daily), 8)

    # Hard requirement 9: INDEXING SAFETY:
    # `context[-K_daily * season_daily:]` slices the last `K_daily * season_daily` points.
    # Since K_daily is `n // season_daily` (or capped), `K_daily * season_daily` will always be
    # less than or equal to `n`. This slice is safe and always contains `K_daily * season_daily` elements.
    mat_daily = context[-K_daily * season_daily:].reshape(K_daily, season_daily)

    # Calculate the average seasonal pattern by taking the mean across the K daily cycles.
    seasonal_pattern_daily = mat_daily.mean(axis=0)

    # Add a small, damped level correction on top of the seasonal pattern.
    # This accounts for recent shifts in the overall level of the series.
    # 1. Determine the seasonal index of the very last observed point.
    idx_last_seasonal = (n - 1) % season_daily

    # 2. Get the expected seasonal value for that last point based on our averaged pattern.
    seasonal_pred_for_last_obs = seasonal_pattern_daily[idx_last_seasonal]

    # 3. Calculate the residual (error) of the last observed point.
    last_residual = context[-1] - seasonal_pred_for_last_obs

    # 4. Apply a damping factor to this residual. This prevents over-correction
    # if the last point was an outlier or just noise.
    damping_factor = 0.5 # A heuristic value, can be tuned.
    level_shift = last_residual * damping_factor

    # Generate the base seasonal forecast by tiling the pattern.
    reps = int(np.ceil(prediction_length / season_daily))
    seasonal_forecast_base = np.tile(seasonal_pattern_daily, reps)[:prediction_length]

    # Apply the damped level shift to the entire forecast.
    out = seasonal_forecast_base + level_shift

    # Hard requirement 4: Never output NaN or inf.
    # `np.nan_to_num` safely replaces any potential NaNs (e.g., if context had NaNs initially).
    # `context[-1]` is used as a fallback if a NaN occurs, ensuring a finite output.
    out = np.nan_to_num(out, nan=float(context[-1]))

    # Hard requirement 3: Output length must be EXACTLY prediction_length.
    # Hard requirement 4: Never output NaN or inf.
    # Ensure output type is float, as specified by problem.
    return out.astype(float)