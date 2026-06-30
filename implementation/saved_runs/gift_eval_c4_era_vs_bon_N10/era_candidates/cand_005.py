import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback to naive forecast for very short contexts (less than 4 points)
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate recent slope and scale for damped trend
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a relevant scale for clipping
    # Use min(n, 13) to ensure we don't index beyond the array for shorter series
    history_for_scale = context[-min(n, 13):]
    scale = float(np.std(history_for_scale))
    
    # If standard deviation is zero (all values are same), use mean absolute value as fallback scale, or 1.0
    if scale == 0:
        scale = np.mean(np.abs(history_for_scale))
        if scale == 0: # If all values are zero, scale defaults to 1.0 to avoid division by zero issues
            scale = 1.0

    # Clip slope to a small fraction of the scale to ensure conservative trend
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damped trend
    phi = 0.6  # Damping factor
    steps = np.arange(1, prediction_length + 1)
    # The cumsum applies the damping to each step's contribution:
    # last + slope*phi + slope*phi^2 + ...
    # This is equivalent to last + slope * (phi + phi^2 + ... + phi^k)
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping based on recent history's min/max
    history_for_bounds = context[-min(n, 13):]
    lo = float(np.min(history_for_bounds))
    hi = float(np.max(history_for_bounds))
    span = hi - lo

    # If recent history is flat (span is 0), use a small margin around the last value
    if span == 0:
        margin = abs(last) * 0.1  # 10% of last value
        if margin == 0: # If last is 0, use a fixed small value
            margin = 1.0
        lo = last - margin
        hi = last + margin
        span = hi - lo # Recalculate span after adjusting lo/hi
    
    # Clip forecasts within a band around recent observations
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Ensure no NaN or infinity values in the output
    # Replace NaNs with 'last', positive infinities with 'hi', negative infinities with 'lo'
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)