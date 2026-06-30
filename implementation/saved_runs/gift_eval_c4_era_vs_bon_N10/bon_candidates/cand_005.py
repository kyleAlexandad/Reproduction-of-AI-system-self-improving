import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short series: return zeros for empty context, last value for very short ones.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback to naive (last-value) forecast for series shorter than a meaningful trend window.
    # The minimum context length to calculate a slope over 3 differences is 4.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a recent, damped trend.
    # Use the last 4 points to compute the average of the last 3 differences.
    # This provides a more stable slope than just the last two points,
    # while still being very recent.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Calculate a characteristic scale of the series for robust clipping of the slope.
    # Use the standard deviation of the last 13 weeks or the entire context if shorter.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # If the series has no variance (scale is 0), set slope to 0 to prevent issues
    # and effectively revert to naive forecast in this case.
    if scale == 0:
        slope = 0.0
    else:
        # Clip the calculated slope to a small fraction (10%) of the series' scale.
        # This prevents aggressive trend extrapolation, ensuring forecasts remain conservative
        # and close to the last observed value, aligning with random-walk-like behavior.
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Damping factor for the trend component.
    # A smaller phi means the trend contribution diminishes faster over the forecast horizon.
    # Using 0.5 for stronger damping, which is more conservative and aligns with the
    # "random walk" nature of the data where trends are not expected to persist long.
    phi = 0.5 

    # Generate an array of steps (1, 2, ..., prediction_length)
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the forecast: last value + a cumulative, damped trend component.
    # The term (phi ** steps) ensures the trend's influence rapidly decreases.
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: confine forecasts within a band around recent observations.
    # Determine the min and max of the last 13 weeks, or the entire context if shorter.
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Calculate the range (span) of recent values.
    span = hi - lo

    # Define clipping bounds: extend the recent min/max range by 25% on both sides.
    # This provides a flexible but bounded range for forecasts.
    # If span is zero (all recent values are identical), clipping bounds become lo/hi themselves.
    if span == 0:
        lo_clip = lo
        hi_clip = hi
    else:
        lo_clip = lo - 0.25 * span
        hi_clip = hi + 0.25 * span
        
    out = np.clip(out, lo_clip, hi_clip)

    # Handle any potential NaN or infinity values that might arise from edge cases (e.g., division by zero,
    # although efforts are made to prevent them).
    # Replace NaN with the last observed value, positive inf with the recent high,
    # and negative inf with the recent low. This ensures valid, conservative outputs.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)