import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard Requirement 8: Handle short context arrays robustly.
    # If context is empty, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # If context is very short, revert to pure naive (last-value) forecast.
    # This prevents issues with calculating meaningful trend/scale from too few points.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a small, damped trend.
    # Use the last 4 observations to determine a recent slope.
    recent_for_slope = context[-min(n, 4):] # Ensure we don't index beyond context
    if len(recent_for_slope) < 2: # Need at least 2 points to calculate diff
        slope = 0.0
    else:
        slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for clipping the slope and for general value bounds.
    # Use the last 13 observations if available, otherwise use all context.
    # This helps in adapting to recent volatility without being too sensitive to old data.
    data_for_scale = context[-min(n, 13):]
    # Handle cases where std dev might be zero (e.g., all values are same)
    scale = float(np.std(data_for_scale))
    if scale == 0:
        # If scale is zero, use a small default or revert to naive-like behavior for slope
        # This prevents division by zero or issues with relative clipping.
        # Forcing slope to zero is a safe fallback.
        slope = 0.0
    else:
        # Heavily damp the slope: cap it to a small fraction of the recent scale.
        # This prevents aggressive trend extrapolation, keeping forecasts close to naive.
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damping factor (phi) to the trend.
    # A value of 0.6 means the trend contribution diminishes quickly over the horizon.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the forecast as the last observed value plus a damped trend.
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations.
    # This provides a guardrail, preventing forecasts from becoming too extreme.
    data_for_bounds = context[-min(n, 13):]
    lo = float(np.min(data_for_bounds))
    hi = float(np.max(data_for_bounds))
    span = hi - lo

    # If all values are identical, span will be 0. Adjust clipping to just be hi/lo.
    if span == 0:
        out = np.clip(out, lo, hi)
    else:
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Hard Requirement 4: Never output NaN or inf.
    # Replace any potential NaNs or Infs that might arise from extreme edge cases.
    # Fallback values are `last` for NaN, `hi` for positive inf, `lo` for negative inf.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)