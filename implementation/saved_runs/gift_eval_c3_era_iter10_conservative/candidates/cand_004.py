import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback for very short contexts: return naive forecast (last value)
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate recent slope (trend) from the last 4 points
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope, using last 13 points or full context
    if n >= 13:
        scale = float(np.std(context[-13:]))
    else:
        scale = float(np.std(context))
    
    # Handle cases where std might be zero (e.g., all values are same)
    if scale == 0:
        slope_clip_bound = 0.0 # No trend if no variance
    else:
        slope_clip_bound = 0.1 * scale
        
    # Conservatively clip the slope to prevent aggressive trend extrapolation
    slope = float(np.clip(slope, -slope_clip_bound, slope_clip_bound))

    # Apply a damped trend extrapolation
    phi = 0.6  # Damping factor
    steps = np.arange(1, prediction_length + 1)
    
    # Initialize output with last value and add damped trend
    out = last + slope * np.cumsum(phi ** steps)

    # Get recent min/max for robust clipping of forecasts
    if n >= 13:
        lo = float(np.min(context[-13:]))
        hi = float(np.max(context[-13:]))
    else:
        lo = float(np.min(context))
        hi = float(np.max(context))
    
    span = hi - lo

    # Robustly clip forecasts within a band around recent observations
    # This prevents forecasts from running away too much
    # If span is 0 (all recent values are the same), this clipping effectively keeps forecasts at 'lo'/'hi'.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Ensure no NaN or Inf in output, replacing with sensible fallbacks
    # nan=last is a safe default for NaNs
    # posinf=hi and neginf=lo clip to the observed recent range for extreme values
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)