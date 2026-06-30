import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback to naive (last value) for very short contexts where trend calculation might be unreliable
    # This prevents noise from very few data points influencing the forecast too much.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # --- Damped Trend Calculation ---
    # Use the last 4 points to calculate a recent slope.
    # This window is short to capture recent changes but not too short to be overly noisy.
    recent_for_slope = context[-min(n, 4):] # Ensure we don't try to slice beyond array length
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for clipping the slope based on recent variability.
    # Using std dev of the last 13 points (or all available if n < 13) provides a robust scale.
    recent_for_scale = context[-min(n, 13):]
    scale = float(np.std(recent_for_scale)) if len(recent_for_scale) > 0 else 0.0

    # Clip the slope to a small fraction (10%) of the recent scale.
    # This is a critical conservative step to prevent trend forecasts from running away.
    if scale == 0: # If all recent values are the same, no trend should be applied.
        slope = 0.0
    else:
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    phi = 0.6  # Damping factor: the trend effect diminishes over time (0.6 is conservative).
    steps = np.arange(1, prediction_length + 1)

    # Apply the damped trend to the last observed value.
    # np.cumsum(phi ** steps) calculates the cumulative sum of damped trend contributions.
    out = last + slope * np.cumsum(phi ** steps)

    # --- Robust Clipping to a Recent Band ---
    # Identify the min/max of recent observations to define a plausible range for forecasts.
    # Using the last 13 points (or all available if n < 13) for this range.
    recent_for_bounds = context[-min(n, 13):]
    if len(recent_for_bounds) > 0:
        lo = float(np.min(recent_for_bounds))
        hi = float(np.max(recent_for_bounds))
        span = hi - lo
        # Clip forecasts within a small margin (25%) around the recent min/max.
        # This prevents extreme values and keeps forecasts "in-band".
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    else:
        # Fallback for extreme cases (should be caught by n=0 check, but safe guard)
        lo, hi = last, last
        out = np.clip(out, lo, hi)

    # --- Final Sanitization ---
    # Replace any potential NaN or infinity values with sane defaults.
    # NaNs become the last observed value, positive inf becomes the recent max (hi),
    # and negative inf becomes the recent min (lo).
    # .astype(float) ensures the final output array's dtype is float.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)