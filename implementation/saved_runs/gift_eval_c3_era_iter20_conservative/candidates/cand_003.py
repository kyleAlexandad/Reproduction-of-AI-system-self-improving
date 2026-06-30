import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short contexts
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    if n < 4:
        # For extremely short series, revert to simple naive (last value)
        return np.full(prediction_length, last, dtype=float)

    # Calculate a short-term trend
    # Use the last 4 points for slope, as recommended for conservative approach
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope and output range
    # Use std dev of last 13 points or whole context if shorter
    scale_window_length = 13
    if n >= scale_window_length:
        scale = float(np.std(context[-scale_window_length:]))
    else:
        scale = float(np.std(context))
    
    # Handle cases where scale might be zero (e.g., all recent values are identical)
    # If scale is zero, slope will be clipped to zero, making it naive.
    if scale == 0:
        scale = 1.0 # Prevent division by zero if we were to normalize, though not directly used that way here.
                    # This ensures clipping bounds are not zero, allowing at least a tiny possible slope.
                    # For this specific clip, if scale is 0, clip to [-0, 0] which is 0, so it's safe without this line.
                    # Keeping it as a defensive measure.

    # Clip the slope to a small fraction of the recent scale to ensure "tiny correction"
    max_slope_factor = 0.1 # Cap slope at +/- 10% of recent std dev
    slope = float(np.clip(slope, -max_slope_factor * scale, max_slope_factor * scale))

    # Apply a damped trend
    phi = 0.6  # Damping factor; smaller phi means trend effect diminishes faster
    steps = np.arange(1, prediction_length + 1)
    
    # The cumulative sum of phi^steps applies increasing, but damped, trend contribution
    forecasts = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations
    clip_window_length = 13
    if n >= clip_window_length:
        lo = float(np.min(context[-clip_window_length:]))
        hi = float(np.max(context[-clip_window_length:]))
    else:
        lo = float(np.min(context))
        hi = float(np.max(context))
    
    span = hi - lo
    
    # Handle cases where span might be zero (e.g., all recent values are identical)
    if span == 0:
        # If all values are the same, lo and hi are the same, span is 0.
        # Clip to this single value, effectively making it naive.
        forecasts = np.clip(forecasts, lo, hi)
    else:
        # Clip forecasts within +/- 25% of the recent range
        clip_margin_factor = 0.25
        forecasts = np.clip(forecasts, lo - clip_margin_factor * span, hi + clip_margin_factor * span)

    # Ensure no NaN or Inf values are returned
    # Replace NaN with last observed value, PosInf with max, NegInf with min
    return np.nan_to_num(forecasts, nan=last, posinf=hi, neginf=lo).astype(float)