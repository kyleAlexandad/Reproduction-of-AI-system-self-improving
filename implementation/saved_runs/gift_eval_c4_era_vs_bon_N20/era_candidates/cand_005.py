import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback for very short series: pure naive forecast
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend from recent values
    # Use last 4 points for slope calculation
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope, using up to the last 13 points
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    if scale == 0: # Avoid division by zero or issues with constant series
        scale = 1.0 # Use a default scale

    # Heavily clip the slope to ensure only tiny, safe corrections
    # Limit slope to 10% of the recent standard deviation
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a strong damping factor (phi <= 0.7 as recommended)
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate forecasts with damped trend
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations
    # Use up to the last 13 points for min/max
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    span = hi - lo
    
    # Add a small margin (e.g., 25% of the recent span) to min/max for clipping
    # This allows forecasts to slightly extend beyond observed range but prevents wild extrapolation
    clip_lo = lo - 0.25 * span
    clip_hi = hi + 0.25 * span
    
    # Special handling for constant series where span is 0
    if span == 0:
        clip_lo = lo - abs(0.1 * last) if last != 0 else -0.1
        clip_hi = hi + abs(0.1 * last) if last != 0 else 0.1
        # Ensure lo < hi if possible, for cases where last is 0 or very small
        if clip_lo >= clip_hi:
            clip_hi = clip_lo + 0.1 # Ensure a small valid range

    out = np.clip(out, clip_lo, clip_hi)

    # Final safety: replace any potential NaN/inf values (unlikely after clipping)
    # with `last` for NaN, and `hi`/`lo` for pos/neg inf respectively.
    # Ensure the output type is float.
    return np.nan_to_num(out, nan=last, posinf=clip_hi, neginf=clip_lo).astype(float)