import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short context arrays:
    # If no data, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    last_value = float(context[-1])

    # If context is very short (e.g., less than 4 points for trend estimation),
    # fallback to simple naive (last value).
    if n < 4:
        return np.full(prediction_length, last_value, dtype=float)

    # Calculate a damped trend for series with sufficient history.
    # Use the last 4 points to estimate a recent slope.
    recent_context = context[-4:]
    slope = float(np.mean(np.diff(recent_context)))

    # Estimate scale for robust capping of the slope.
    # Use standard deviation of the last 13 points if available, otherwise of the whole context.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Ensure scale is not zero to prevent division by zero or NaN issues in future steps if it were used in division.
    # While not directly used in division here, it's a good practice for robustness.
    # For now, it's used for clipping, so a small non-zero value is fine if std is 0.
    if scale == 0:
        # If all recent values are identical, treat scale as a small epsilon
        # or base it on a small fraction of the last_value if last_value is non-zero,
        # otherwise use a tiny constant.
        scale = np.fmax(1e-6, np.abs(last_value) * 0.01)


    # Clip the slope to a small fraction of the recent scale to prevent aggressive extrapolation.
    # This is a critical conservative step.
    max_slope_change = 0.1 * scale
    slope = float(np.clip(slope, -max_slope_change, max_slope_change))

    # Apply a damping factor (phi) to the trend.
    # A value of 0.6 means the trend's influence diminishes quickly over the forecast horizon.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the forecast: last value plus a damped cumulative trend.
    forecasts = last_value + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a sensible band around recent observations.
    # Determine the min/max of the last 13 points or the entire context.
    min_val_recent = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    max_val_recent = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    span_recent = max_val_recent - min_val_recent
    
    # Define the clipping bounds: recent min/max +/- 25% of the recent span.
    # This prevents forecasts from going too far outside the historical range.
    lower_bound = min_val_recent - 0.25 * span_recent
    upper_bound = max_val_recent + 0.25 * span_recent

    # If span is zero (all recent values are the same), adjust bounds to be around the last value.
    if span_recent == 0:
        # Use a small arbitrary range around the last value if context is flat
        lower_bound = last_value - np.fmax(1e-6, np.abs(last_value) * 0.05)
        upper_bound = last_value + np.fmax(1e-6, np.abs(last_value) * 0.05)


    forecasts = np.clip(forecasts, lower_bound, upper_bound)

    # Ensure no NaN or Inf values are returned. Replace them with sane defaults.
    # nan is replaced by the last observed value.
    # posinf/neginf are replaced by the upper/lower bounds from clipping.
    final_forecasts = np.nan_to_num(
        forecasts,
        nan=last_value,
        posinf=upper_bound,
        neginf=lower_bound
    ).astype(float)

    return final_forecasts