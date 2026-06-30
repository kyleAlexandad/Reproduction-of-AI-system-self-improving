import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback for very short series (less than 4 points): Naive forecast (repeat last value)
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a small, damped trend
    # Use the last 4 observations to estimate the slope
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for clipping the slope and for range bounding
    # Use the last 13 observations if available, otherwise the entire context
    if n >= 13:
        scale = float(np.std(context[-13:]))
        lo = float(np.min(context[-13:]))
        hi = float(np.max(context[-13:]))
    else:
        scale = float(np.std(context))
        lo = float(np.min(context))
        hi = float(np.max(context))
    
    # Handle cases where scale might be zero (e.g., all recent values are the same)
    if scale == 0:
        scale = abs(last) if abs(last) > 1e-6 else 1.0 # Use absolute last value as scale if zero

    # Clip the slope to a small fraction of the recent scale
    # This ensures the trend correction is always minor
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Damping factor for the trend, recommended to be <= 0.7
    phi = 0.6

    # Generate steps for the prediction horizon
    steps = np.arange(1, prediction_length + 1)

    # Calculate forecasts: last value + damped cumulative trend
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: ensure forecasts stay within a reasonable band
    # The band is based on the recent min/max, extended by a small margin
    span = hi - lo
    
    # Handle cases where span might be zero (e.g., all recent values are the same)
    if span == 0:
        # If all values are the same, lo and hi are equal. Make a small span for clipping.
        # This keeps the forecast fixed at 'last' if the trend is also zero.
        # If trend is non-zero but span is zero, it allows slight deviation but still clips.
        clip_margin = abs(last) * 0.05 # A small percentage of the last value
        if clip_margin < 1e-6: # Ensure a minimum margin
            clip_margin = 0.1
        out = np.clip(out, last - clip_margin, last + clip_margin)
    else:
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Final robust handling of NaN/inf values and ensure float type output
    # If any NaNs appear (shouldn't with robust clipping, but as a safeguard), replace with 'last'.
    # If posinf/neginf (again, unlikely), clip to 'hi'/'lo'.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)