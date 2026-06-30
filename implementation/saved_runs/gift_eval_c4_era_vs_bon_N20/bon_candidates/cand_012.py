import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short contexts (including empty)
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last_value = float(context[-1])

    # For contexts shorter than required for trend calculation, fallback to naive
    if n < 4:
        return np.full(prediction_length, last_value, dtype=float)

    # Calculate recent trend (slope)
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope and for forecast bounds
    # Use standard deviation of the last 13 points if available, otherwise all context
    if n >= 13:
        scale = float(np.std(context[-13:]))
        lo_bound_context = context[-13:]
    else:
        scale = float(np.std(context))
        lo_bound_context = context

    # If scale is zero (e.g., flat series), avoid division by zero and make slope zero
    if scale == 0:
        clipped_slope = 0.0
    else:
        # Clip the slope to a small fraction of the scale to prevent forecasts from running away
        max_slope_change = 0.1 * scale
        clipped_slope = float(np.clip(slope, -max_slope_change, max_slope_change))

    # Damping factor for the trend
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)

    # Calculate forecasts with damped trend
    # The cumulative sum of phi**steps ensures the trend effect diminishes over time
    forecasts = last_value + clipped_slope * np.cumsum(phi ** steps)

    # Determine robust clipping bounds from recent observations
    lo = float(np.min(lo_bound_context))
    hi = float(np.max(lo_bound_context))
    span = hi - lo

    # If span is zero (flat series), define a small default span to avoid issues
    if span == 0:
        # If the series is perfectly flat, lo and hi are the same.
        # Create a small artificial span to allow some clipping margin,
        # but ensure forecasts stay at the flat value unless a slope pushes them.
        # This fallback makes sense if the series is e.g. [10, 10, 10]
        # In such a case, the slope would be 0 and forecasts would be 10,
        # so clipping around 10 +/- epsilon is harmless and safe.
        epsilon = 1e-6 # A tiny value for flat series
        lo_clip = lo - epsilon
        hi_clip = hi + epsilon
    else:
        # Extend the min/max range by a small margin for robust clipping
        lo_clip = lo - 0.25 * span
        hi_clip = hi + 0.25 * span
    
    # Clip forecasts to stay within a reasonable band around recent observations
    forecasts = np.clip(forecasts, lo_clip, hi_clip)

    # Ensure no NaN or Inf values are returned. Replace with safe defaults.
    # nan replaced with last_value, posinf with hi_clip, neginf with lo_clip.
    return np.nan_to_num(forecasts, nan=last_value, posinf=hi_clip, neginf=lo_clip).astype(float)