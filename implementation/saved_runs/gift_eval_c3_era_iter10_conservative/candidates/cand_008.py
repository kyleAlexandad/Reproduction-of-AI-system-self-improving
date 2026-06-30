import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context by returning all zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Handle very short contexts (less than 4 points) by returning naive (last value) forecast.
    # This also ensures 'recent = context[-4:]' is safe as it's only accessed when n >= 4.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate raw trend (slope) from the last 4 points.
    # This provides a very short-term view of recent direction.
    recent = context[-4:]
    raw_slope = float(np.mean(np.diff(recent)))

    # Determine a scale for damping and clipping.
    # Use the last 13 points if available, otherwise use all available context.
    scale_context = context[-13:] if n >= 13 else context
    scale = float(np.std(scale_context))

    # Clip the raw slope to a small fraction of the recent standard deviation.
    # This prevents the trend from being too aggressive. If scale is 0 (constant series),
    # the clip bounds become [-0, 0], so clipped_slope will correctly be 0.
    clipped_slope = float(np.clip(raw_slope, -0.1 * scale, 0.1 * scale))

    # Apply an explicit blend weight to the trend component.
    # This effectively creates an ensemble where Naive (last value) has a high weight,
    # and the damped trend contributes a smaller, controlled portion.
    # For example, 0.25 means 75% Naive + 25% Damped Trend.
    trend_blend_weight = 0.25 # Reduced from implicit 1.0 in parent
    effective_slope = clipped_slope * trend_blend_weight

    # Damping factor for the trend contribution over the forecast horizon.
    # Reduced 'phi' makes the trend decay even faster over time, aligning with
    # the "random walk" behavior and conservatism recommended for this task.
    phi = 0.5 # Parent used 0.6, reducing this makes trend fade faster

    # Steps for the forecast horizon (1 to prediction_length)
    steps = np.arange(1, prediction_length + 1)

    # Calculate forecasts. The cumulative sum of phi**steps ensures the trend effect
    # diminishes significantly over the forecast horizon.
    out = last + effective_slope * np.cumsum(phi ** steps)

    # Robust clipping of forecasts based on recent historical min/max.
    # This keeps forecasts within a plausible range derived from recent observations.
    clip_context = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_context))
    hi = float(np.max(clip_context))
    span = hi - lo

    # If the history is constant (span is 0), then lo == hi == last.
    # The clipping band becomes [last, last], effectively ensuring forecasts
    # remain 'last' for constant series, which is correct.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Ensure no NaN or inf values are returned.
    # Fallback to 'last' for NaN, and historical 'hi'/'lo' for inf, providing robustness.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)