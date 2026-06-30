import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short context arrays
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback to naive forecast for series too short to estimate trend/scale reliably
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a small, damped trend
    # Use the last 4 points for slope estimation
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Estimate scale for clipping the slope and forecasting bounds
    # Use last 13 points if available, otherwise the whole context
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    # Guard against zero scale in case context values are all the same
    if scale == 0:
        scale = np.mean(np.abs(context[-13:])) if n >= 13 else np.mean(np.abs(context))
        if scale == 0: # If still zero, use a small epsilon
            scale = 1.0 # Or use a more robust fallback like absolute mean of recent values, or a fixed small value
    
    # Clip the slope to a small fraction of the scale to prevent aggressive extrapolation
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply damping factor (phi <= 0.7 recommended)
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Generate initial forecasts using last value + damped trend
    out = last + slope * np.cumsum(phi ** steps)

    # Apply robust clipping to keep forecasts within a reasonable band
    # Use recent min/max from last 13 points, or entire context if shorter
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    span = hi - lo

    # Handle cases where all context values are identical (span is 0)
    if span == 0:
        # If span is zero, create a small band around the last value
        lo_clip = last - 0.05 * np.abs(last) if last != 0 else -0.05
        hi_clip = last + 0.05 * np.abs(last) if last != 0 else 0.05
        if lo_clip == hi_clip: # Ensure distinct boundaries for clipping
            lo_clip -= 0.01
            hi_clip += 0.01
    else:
        # Extend the min/max range by a small margin
        lo_clip = lo - 0.25 * span
        hi_clip = hi + 0.25 * span
    
    out = np.clip(out, lo_clip, hi_clip)

    # Ensure no NaN or Inf in the output. Fallback to `last` for NaN, `hi_clip` for posinf, `lo_clip` for neginf.
    # We use hi_clip and lo_clip here to ensure consistency with the clipping range.
    return np.nan_to_num(out, nan=last, posinf=hi_clip, neginf=lo_clip).astype(float)