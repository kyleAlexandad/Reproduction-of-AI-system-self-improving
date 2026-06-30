import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short contexts
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    last = float(context[-1])

    if n < 4:  # If too few points for trend calculation, default to Naive
        return np.full(prediction_length, last, dtype=float)

    # Calculate a conservative, damped trend
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope, using a wider window if available
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Handle cases where scale might be zero (e.g., flat series) to prevent division by zero implicitly
    if scale == 0:
        slope_clip_max = 0.0
    else:
        slope_clip_max = 0.1 * scale # Limit slope to a small fraction of the data's standard deviation
        
    slope = float(np.clip(slope, -slope_clip_max, slope_clip_max))

    # Apply a strong damping factor to the trend contribution
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate forecasts with damped trend
    out = last + slope * np.cumsum(phi ** steps)

    # Clip forecasts to a conservative band around recent observations
    # Use wider window for min/max if available
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    span = hi - lo
    
    # Handle cases where span might be zero (e.g., flat series)
    if span == 0:
        # If span is zero, all recent values are the same. Clip to that value.
        out = np.clip(out, lo, hi)
    else:
        # Expand the band by a small margin (25% of span)
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Ensure no NaN or Inf values are returned
    # Fallback to 'last' for NaN, 'hi' for posinf, 'lo' for neginf
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)