import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short series
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    last = float(context[-1])
    
    # If context is too short for trend calculation, fallback to pure naive
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a conservative, damped trend
    recent = context[-4:] # Use last 4 points for slope calculation
    
    # Calculate raw slope as mean of differences
    slope = float(np.mean(np.diff(recent)))
    
    # Determine a scale for clipping the slope
    # Use standard deviation of last 13 points, or full context if shorter
    scale = float(np.std(context[-13:]) if n >= 13 else np.std(context))
    
    # Clip the slope to a small fraction of the scale to prevent over-extrapolation
    # This ensures the trend correction is very minor. If scale is 0 (all values same),
    # slope will be clipped to 0.
    if scale == 0: # Avoid division by zero and ensure slope is zero if no variability
        slope = 0.0
    else:
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damped trend
    phi = 0.6  # Damping factor (0.6 is quite aggressive damping, so trend fades quickly)
    steps = np.arange(1, prediction_length + 1)
    
    # Forecast by adding the damped trend to the last observed value
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations
    # Get min/max from last 13 points, or full context if shorter
    lo = float(np.min(context[-13:]) if n >= 13 else np.min(context))
    hi = float(np.max(context[-13:]) if n >= 13 else np.max(context))
    
    span = hi - lo
    
    # If span is 0 (all recent values are the same), create a small non-zero span to avoid issues
    # and ensure clipping bounds are not identical.
    if span == 0:
        span = abs(last) * 0.1 if abs(last) > 1e-6 else 1.0 # Use 10% of last value, or 1.0 if last is near zero
        lo = last - 0.5 * span
        hi = last + 0.5 * span

    # Clip output to a range around the recent observations min/max
    # Adding a small margin (0.25 * span) to allow for slight movements
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Ensure no NaN or inf values in the output
    # Fallback to 'last' for NaN, 'hi' for posinf, 'lo' for neginf
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)
