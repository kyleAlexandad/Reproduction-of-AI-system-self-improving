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
    
    # Calculate standard deviation. If it's zero (constant series), no trend should be applied.
    scale = float(np.std(scale_context))
    
    clipped_slope = 0.0
    if scale > 0: # Only apply trend logic if there is variability in the data
        # Clip the raw slope to an even smaller fraction of the recent standard deviation.
        # This makes the trend extremely conservative. Reduced from 0.1 to 0.05.
        clip_factor = 0.05
        clipped_slope = float(np.clip(raw_slope, -clip_factor * scale, clip_factor * scale))

    # Apply an even smaller blend weight to the trend component.
    # This means the forecast is significantly closer to pure Naive (last value).
    # Reduced from 0.25 to 0.10, effectively making it 90% naive, 10% damped trend.
    trend_blend_weight = 0.10
    effective_slope = clipped_slope * trend_blend_weight

    # Damping factor for the trend contribution over the forecast horizon.
    # Reduced 'phi' makes the trend decay much faster over time, becoming negligible quickly,
    # aligning with the "random walk" behavior of the series. Reduced from 0.5 to 0.3.
    phi = 0.3

    # Steps for the forecast horizon (1 to prediction_length)
    steps = np.arange(1, prediction_length + 1)

    # Calculate forecasts. The cumulative sum of phi**steps ensures the trend effect
    # diminishes significantly over the forecast horizon.
    out = last + effective_slope * np.cumsum(phi ** steps)

    # Robust clipping of forecasts based on recent historical min/max.
    # This keeps forecasts within an even tighter plausible range derived from recent observations.
    clip_context = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_context))
    hi = float(np.max(clip_context))
    span = hi - lo

    # If the history is constant (span is 0), then lo == hi == last.
    # The clipping band becomes [last, last], effectively ensuring forecasts
    # remain 'last' for constant series, which is correct.
    # Reduced clipping margin from 0.25 to 0.1 for tighter bounds.
    out = np.clip(out, lo - 0.1 * span, hi + 0.1 * span)

    # Ensure no NaN or inf values are returned.
    # Fallback to 'last' for NaN, and historical 'hi'/'lo' for inf, providing robustness.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)