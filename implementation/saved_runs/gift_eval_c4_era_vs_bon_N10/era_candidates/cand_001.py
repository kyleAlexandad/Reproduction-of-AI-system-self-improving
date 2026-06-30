import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Handle very short context (less than 4 points): fallback to Naive (last-value)
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend for forecasts
    # Slope based on the last 4 points to capture recent direction
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope and output, using last 13 points or all available
    # This prevents the slope from being too aggressive relative to recent variability
    scale = float(np.std(context[-13:]) if n >= 13 else np.std(context))

    # If there's no variability (scale is zero), force slope to zero
    if scale == 0:
        slope = 0.0
    else:
        # Clip the slope to a small fraction of the scale to ensure conservatism
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Damping factor for the trend, ensures it fades out
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)

    # Generate forecasts by applying the damped trend to the last observed value
    out = last + slope * np.cumsum(phi ** steps)

    # Robust Clipping: keep forecasts within a reasonable band around recent observations
    # Define bounds based on min/max of the last 13 points or all available
    lo = float(np.min(context[-13:]) if n >= 13 else np.min(context))
    hi = float(np.max(context[-13:]) if n >= 13 else np.max(context))
    span = hi - lo

    # If the span is zero (all recent values are the same), force forecasts to that value
    if span == 0:
        out = np.full(prediction_length, lo, dtype=float)
    else:
        # Extend the min/max range slightly and clip forecasts
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Ensure no NaN or Inf values are returned. Replace them with safe defaults.
    out = np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)

    return out