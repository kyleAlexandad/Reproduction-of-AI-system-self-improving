import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short series
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    if n < 4:
        # Fallback to naive forecast for very short series
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend component
    # Use the last 4 points for slope calculation, as suggested by the template
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Calculate scale for clipping the slope and for range-based clipping
    # Use context[-13:] if available, otherwise the full context.
    if n >= 13:
        context_for_scale_and_clip = context[-13:]
    else:
        context_for_scale_and_clip = context

    # Ensure context_for_scale_and_clip has at least 2 elements for std,
    # otherwise std will be 0 or NaN. If only one element, std is 0.
    if len(context_for_scale_and_clip) > 1:
        scale = float(np.std(context_for_scale_and_clip))
    else:
        scale = 0.0 # If only one element, std is 0.

    # If scale is zero (e.g., all recent values are identical), use a small default to avoid issues.
    if scale < 1e-6:
        # A conservative fallback: use a fraction of the last value or a default small value
        if abs(last) > 1e-6:
            scale = abs(last) * 0.1
        else:
            scale = 1.0 # Default scale if last is also near zero

    # Cap the slope to a small fraction of the scale to prevent aggressive extrapolation
    max_slope_change = 0.1 * scale
    slope = float(np.clip(slope, -max_slope_change, max_slope_change))

    # Damping factor and forecast steps for the damped trend
    phi = 0.6 # As suggested in the template
    steps = np.arange(1, prediction_length + 1)

    # Generate initial forecasts using the last value and damped trend
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations
    lo = float(np.min(context_for_scale_and_clip))
    hi = float(np.max(context_for_scale_and_clip))
    span = hi - lo

    # If span is zero (all values in recent_for_clip are identical), adjust clipping range
    if span < 1e-6:
        # Create a small artificial span around the constant value
        if abs(lo) > 1e-6:
            span = abs(lo) * 0.1
        else:
            span = 1.0 # Default span if `lo` is also near zero
        lo_clip = lo - 0.25 * span
        hi_clip = hi + 0.25 * span
    else:
        lo_clip = lo - 0.25 * span
        hi_clip = hi + 0.25 * span

    # Apply clipping to the forecasts
    out = np.clip(out, lo_clip, hi_clip)

    # Final check for NaNs/Infs and convert to float
    # If any NaN occurred, use 'last' as fallback.
    # For inf, clip to 'hi'/'lo' (which are from the context and should be finite).
    return np.nan_to_num(out, nan=last, posinf=hi_clip, neginf=lo_clip).astype(float)