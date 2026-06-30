import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short or empty contexts
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    last_value = float(context[-1])
    if n < 4:  # For very short series, fall back to Naive (last value)
        return np.full(prediction_length, last_value, dtype=float)

    # Calculate recent trend (slope)
    # Using the last 4 points for a stable, short-term slope
    recent_context_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_context_for_slope)))

    # Determine a scale for damping and clipping. Use recent 13 points or all context if shorter.
    scale_context = context[-13:] if n >= 13 else context
    # Handle cases where context might be constant, leading to std=0
    scale = float(np.std(scale_context))
    if scale == 0:
        # If all values are the same, scale should be a tiny positive value to avoid division by zero
        # and allow for a small relative margin for clipping if needed.
        # However, for slope damping, a zero scale naturally clips slope to zero, which is desired.
        scale = np.abs(last_value) * 0.1 + 1e-6 # use a small value relative to last_value, or a tiny constant

    # Damp the slope significantly to make it a "tiny correction"
    # Clip the slope to a small fraction of the recent scale
    slope = np.clip(slope, -0.1 * scale, 0.1 * scale).astype(float)

    # Damping factor for the trend contribution over the forecast horizon
    phi = 0.6
    # Steps for the forecast horizon
    steps = np.arange(1, prediction_length + 1)

    # Calculate forecasts with damped trend
    # The cumulative sum of phi^steps ensures the trend effect diminishes over time
    forecast_values = last_value + slope * np.cumsum(phi ** steps)

    # Robust clipping based on recent historical min/max
    clip_context = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_context))
    hi = float(np.max(clip_context))
    span = hi - lo

    # Adjust clipping bounds, handle cases where history might be flat (span=0)
    if span == 0:
        # If all historical values are the same, clip forecasts to be very close to that value
        margin = np.abs(last_value) * 0.05 + 1e-6
        forecast_values = np.clip(forecast_values, last_value - margin, last_value + margin)
    else:
        # Clip forecasts within a band around recent observations (lo, hi)
        forecast_values = np.clip(forecast_values, lo - 0.25 * span, hi + 0.25 * span)

    # Ensure no NaN or inf values are returned
    # Fallback to last_value for NaN, and historical hi/lo for inf
    forecast_values = np.nan_to_num(forecast_values, nan=last_value, posinf=hi, neginf=lo)

    return forecast_values.astype(float)