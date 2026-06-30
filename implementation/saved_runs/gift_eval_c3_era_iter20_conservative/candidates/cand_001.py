import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle SHORT context arrays robustly
    if n == 0:
        # Hard requirement 4: Never output NaN or inf. np.zeros is safe.
        # Output length must be EXACTLY prediction_length.
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback for very short series (n=1, 2, 3) where trend calculation is unstable or meaningless.
    # This also covers the minimum length needed for `recent = context[-4:]`.
    if n < 4:
        # Return naive (last value) forecast.
        return np.full(prediction_length, last, dtype=float)

    # Calculate slope for a damped trend component.
    # Use the last 4 points for a recent and stable slope estimate.
    # Guaranteed n >= 4 here, so context[-4:] is safe.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine the scale of recent fluctuations to clip the slope and output.
    # Use the last 13 points or the full context if shorter (guaranteed n >= 4).
    scale_window = context[-13:] if n >= 13 else context
    
    # Calculate standard deviation. Need at least 2 points for meaningful std.
    # If all values in scale_window are identical, std will be 0.
    if len(scale_window) > 1:
        scale = float(np.std(scale_window))
    else:
        # This case generally shouldn't happen here as n >= 4 ensures len(scale_window) >= 4.
        # However, as a safeguard, if somehow it was 0 or 1, scale is 0.
        scale = 0.0

    # Critically clip the slope to prevent forecasts from running away.
    # This is a key conservative measure, limiting the per-step change to 10% of the recent scale.
    # If scale is 0 (i.e., recent values are flat), this clips slope to 0, making it effectively naive.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damping factor for the trend, ensuring its influence fades over the horizon.
    # Phi = 0.6 is a conservative choice (less than 1).
    phi = 0.6
    
    # Generate the sequence of steps for the prediction horizon (1 to prediction_length).
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the damped trend contribution for each step.
    # np.cumsum(phi ** steps) applies cumulative sum of damped factors for a typical damped trend.
    damped_trend_contribution = slope * np.cumsum(phi ** steps)
    
    # Initial forecast is the last observed value plus the damped trend.
    out = last + damped_trend_contribution

    # Robust clipping: keep forecasts within a band around recent observations.
    # Use the same window as for scale (last 13 points or full context) to determine min/max range.
    clip_window = context[-13:] if n >= 13 else context
    
    lo = float(np.min(clip_window))
    hi = float(np.max(clip_window))
    span = hi - lo

    # Clip the forecasts to a range defined by recent min/max, plus a small margin.
    # If span is 0 (flat recent data), the clipping range becomes [lo, hi], which means [last, last],
    # effectively forcing the forecast to 'last' in such cases.
    # The 0.25 * span margin allows some movement slightly outside the strict recent range.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Hard requirement 4: Never output NaN or inf.
    # `nan=last` is a safe fallback for any NaNs that might have been introduced.
    # `posinf=hi`, `neginf=lo` replace infinite values with reasonable bounds from recent data.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)