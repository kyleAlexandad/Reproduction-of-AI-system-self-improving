import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    # Case 1: Empty context
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last_val = float(context[-1])

    # Case 2: Very short context (e.g., less than 4 points for slope calculation)
    # Revert to naive forecast, which is the strong baseline for this task.
    # The template uses n < 4, which is a good threshold for `context[-4:]`.
    if n < 4:
        return np.full(prediction_length, last_val, dtype=float)

    # Calculate damped trend
    # Use the last 4 points for slope estimation, as per template.
    # This is safe because n is guaranteed to be >= 4 at this point.
    recent_for_slope = context[-4:]
    slope = np.mean(np.diff(recent_for_slope))

    # Calculate a scale to normalize and clip the slope.
    # Use last 13 points if available, otherwise use all available context.
    # This handles shorter series robustly for scale calculation.
    scale_window_len = min(n, 13)
    # np.std of a single element array is 0.0, which is handled correctly by the clipping.
    scale = float(np.std(context[-scale_window_len:]))

    # Clip the slope to be a small fraction of the data's recent scale.
    # This prevents aggressive trend extrapolation, keeping the forecast conservative.
    max_slope_change_factor = 0.1 # As used in the template
    slope = float(np.clip(slope, -max_slope_change_factor * scale, max_slope_change_factor * scale))

    # Damping factor for the trend. Phi=0.6 is from the template, providing moderate damping.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the cumulative damped trend for each step
    # out[h] = last_val + slope * (phi + phi^2 + ... + phi^h)
    forecast_values = last_val + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations.
    # Use last 13 points for min/max, or all context if shorter.
    # This is safe because n is guaranteed to be >= 4.
    clip_data_window_len = min(n, 13)
    clipping_data = context[-clip_data_window_len:]
    lo_ctx = float(np.min(clipping_data))
    hi_ctx = float(np.max(clipping_data))
    span_ctx = hi_ctx - lo_ctx

    # Define clipping bounds based on recent min/max with a margin.
    # If span_ctx is zero (e.g., all recent values are identical),
    # then lo_clip_bound = lo_ctx and hi_clip_bound = hi_ctx.
    # This implies no additional margin for perfectly flat series, which is conservative.
    margin_factor = 0.25 # As used in the template
    lo_clip_bound = lo_ctx - margin_factor * span_ctx
    hi_clip_bound = hi_ctx + margin_factor * span_ctx

    forecast_values = np.clip(forecast_values, lo_clip_bound, hi_clip_bound)

    # Hard requirement 4: Never output NaN or inf.
    # Use nan_to_num to replace any NaN or inf values with sensible defaults.
    # Using the calculated clipping bounds as replacement for inf values ensures consistency.
    final_forecast = np.nan_to_num(forecast_values, nan=last_val, posinf=hi_clip_bound, neginf=lo_clip_bound).astype(float)

    # Hard requirement 3: Output length must be EXACTLY prediction_length.
    # The `steps` array is of length `prediction_length`, so `final_forecast` will have the correct length.
    return final_forecast