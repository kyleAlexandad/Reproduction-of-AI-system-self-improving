import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Handle very short context arrays: return naive last value
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a small, damped trend based on recent observations
    # Use the last 4 points for slope calculation
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Calculate a scale for clipping the slope. Use last 13 points if available,
    # otherwise use the entire context. This prevents extreme slope values.
    scale_context = context[-13:] if n >= 13 else context
    # Handle cases where scale_context might have zero std (e.g., all values are same)
    scale = float(np.std(scale_context))
    if scale == 0: # If there's no variance, trend must be 0
        slope = 0.0
    else:
        # Clip the slope to a small fraction of the recent scale to ensure conservatism
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damping factor to the trend, so it fades over the prediction horizon
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate forecasts as last observed value plus damped trend
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations
    # Determine the recent min/max using the last 13 points or the entire context
    clip_context = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_context))
    hi = float(np.max(clip_context))
    span = hi - lo

    # If span is zero (e.g., all values are same), ensure clipping doesn't expand the range
    if span == 0:
        out = np.clip(out, lo, hi)
    else:
        # Clip the output to a conservative band: recent min/max +/- 25% of the span
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Handle any potential NaN or infinity values, replacing them with safe defaults
    # nan: use the last observed value
    # posinf: clip to the recent maximum
    # neginf: clip to the recent minimum
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)