import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short context arrays
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback for very short series (e.g., 1, 2, 3 points)
    # In such cases, a trend calculation might be unstable or misleading.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend
    # Use the last 4 points for slope calculation for a recent, stable estimate.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope based on recent variability.
    # Use the last 13 points if available, otherwise the entire context.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Ensure scale is not zero to prevent division by zero or nonsensical clipping
    # if all recent values are identical. If scale is 0, slope should be 0.
    if scale == 0:
        slope = 0.0
    else:
        # Clip the slope to allow only tiny corrections relative to recent variability.
        # This prevents the forecast from running away.
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Damping factor for the trend.
    # A value of 0.6 is conservative, as recommended (phi <= 0.7).
    phi = 0.6
    
    # Generate an array for the steps into the future (1 to prediction_length).
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the forecasts: last value + damped cumulative trend.
    # The trend effect diminishes as the steps increase due to phi ** steps.
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping to keep forecasts within a reasonable band around recent observations.
    # Use the last 13 points if available, otherwise the entire context for min/max.
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Calculate the span of recent values.
    span = hi - lo
    
    # Clip forecasts: ensure they stay within recent min/max +/- a small margin.
    # This prevents extreme forecasts, especially if the trend is slightly off.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Final guard: replace any potential NaN or Inf values with sane defaults.
    # NaN values are replaced by the last observed value.
    # Positive infinity is replaced by the recent maximum (hi).
    # Negative infinity is replaced by the recent minimum (lo).
    out = np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)
    
    return out