import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    # If context is empty, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback for very short series (less than 4 points for stable slope calculation)
    # In these cases, a pure naive forecast is safest and sufficient.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend, as recommended for small, safe corrections.
    # Use the last 4 points to determine the recent slope.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Calculate a scale (standard deviation) based on recent observations
    # (last 13 points, or all available if n < 13, which is handled robustly by NumPy slicing).
    scale_window = context[-13:]
    scale = float(np.std(scale_window))

    # Clip the calculated slope to ensure it's a very small correction, relative to the series' scale.
    if scale == 0:
        slope = 0.0  # If the series is flat, there's no trend.
    else:
        # Clip slope to +/- 10% of the standard deviation of the recent window.
        slope = np.clip(slope, -0.1 * scale, 0.1 * scale)

    phi = 0.6  # Damping factor (recommended to be <= 0.7 for strong damping).
    steps = np.arange(1, prediction_length + 1)
    
    # Generate initial forecasts by applying the damped trend to the last observed value.
    out = last + slope * np.cumsum(phi ** steps)

    # Apply robust clipping to keep forecasts within a sensible band around recent observations.
    # Use the same 13-point window (or full context if shorter) for min/max bounds.
    clip_window = context[-13:]
    lo = float(np.min(clip_window))
    hi = float(np.max(clip_window))
    span = hi - lo

    # Adjust clipping bounds:
    if span == 0:
        # If the recent window is flat (min == max), forecasts should stay at that value.
        out = np.clip(out, lo, hi)
    else:
        # Otherwise, clip within recent min/max +/- 25% of the span as a margin.
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Hard requirement 4: Never output NaN or inf.
    # Use nan_to_num for robustness, replacing NaNs with 'last', posinf with 'hi', neginf with 'lo'.
    # Ensure the output array is of type float.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)