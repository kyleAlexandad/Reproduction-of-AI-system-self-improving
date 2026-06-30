import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short contexts (including empty)
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    last = float(context[-1])

    if n < 4:
        # For very short series, fall back to simple last-value naive forecast
        return np.full(prediction_length, last, dtype=float)

    # Calculate a conservative, damped trend
    # Use the last 4 points for slope calculation
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope, using the last 13 points or all if shorter
    scale_context = context[-13:] if n >= 13 else context
    scale = float(np.std(scale_context))

    # Guard against zero scale (e.g., constant series)
    if scale == 0:
        # If series is constant, slope must be 0
        slope = 0.0
    else:
        # Clip the slope to a small fraction of the scale to prevent aggressive extrapolation
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply exponential damping to the trend
    phi = 0.6  # Damping factor, heavily dampens the trend influence over time
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the forecasts
    # last value + cumulative damped slope contribution
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations
    # Use the last 13 points or all available points for min/max
    clip_context = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_context))
    hi = float(np.max(clip_context))
    span = hi - lo

    # Guard against zero span (e.g., constant series)
    if span == 0:
        # If series is constant, forecasts should be just the last value
        out = np.full(prediction_length, last, dtype=float)
    else:
        # Clip forecasts to a band slightly wider than recent min/max
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final safety: replace any NaN or inf values
    # NaN replaced by 'last', posinf by 'hi', neginf by 'lo'
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)