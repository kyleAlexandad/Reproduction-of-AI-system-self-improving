import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short contexts
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # For contexts shorter than 4 points, default to naive (last value)
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a small, damped trend
    # Use the last 4 points for slope calculation for stability
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for clipping the slope, using up to the last 13 points
    # or the entire context if shorter than 13
    scale_period = 13
    if n >= scale_period:
        scale = float(np.std(context[-scale_period:]))
    else:
        scale = float(np.std(context))

    # If scale is zero (e.g., all recent values are the same), prevent division by zero
    # and default to a very small scale to allow some movement if needed, or simply clip to zero range.
    if scale == 0:
        max_abs_val = np.max(np.abs(context))
        if max_abs_val > 0:
            scale = max_abs_val * 0.01 # A small fraction of the current magnitude
        else:
            scale = 1.0 # Default to 1 if context is all zeros

    # Clip the slope to a small fraction of the recent scale to ensure it's a tiny correction
    # This prevents the forecast from running away
    max_slope_factor = 0.1
    slope = float(np.clip(slope, -max_slope_factor * scale, max_slope_factor * scale))

    # Damping factor for the trend. A value of 0.6 means the trend contribution
    # decays fairly quickly.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)

    # Generate forecasts: last value + damped trend
    out = last + slope * np.cumsum(phi ** steps)

    # Apply robust clipping based on recent observations
    # Use up to the last 13 points for min/max
    clip_period = 13
    if n >= clip_period:
        lo = float(np.min(context[-clip_period:]))
        hi = float(np.max(context[-clip_period:]))
    else:
        lo = float(np.min(context))
        hi = float(np.max(context))

    # Add a small margin to the min/max range for clipping
    span = hi - lo
    # If span is zero (e.g., all recent values are the same), create a small artificial span
    if span == 0:
        if last == 0: # If all zeros, allow slight deviation
            lo = -1.0
            hi = 1.0
        else: # Otherwise, allow a small percentage deviation around the last value
            lo = last * 0.9
            hi = last * 1.1
        span = hi - lo # Recalculate span after adjusting lo/hi

    margin_factor = 0.25
    out = np.clip(out, lo - margin_factor * span, hi + margin_factor * span)

    # Ensure no NaN or Inf values are returned
    # Replace NaN with the last observed value, Inf with hi, -Inf with lo
    final_forecast = np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)

    return final_forecast