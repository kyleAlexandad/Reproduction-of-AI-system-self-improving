import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    last = float(context[-1])

    # Handle very short context arrays (n < 4).
    # For these series, fall back to a pure naive (last value) forecast,
    # but still apply robust clipping and NaN/Inf handling for safety.
    if n < 4:
        out = np.full(prediction_length, last, dtype=float)
        
        # Determine min/max from the available short context for clipping bounds.
        # If n=1, lo=hi=last, span=0, so clipping effectively does nothing.
        lo_clip = float(np.min(context))
        hi_clip = float(np.max(context))
        span_clip = hi_clip - lo_clip
        
        # Apply robust clipping to the naive forecasts.
        out_clipped = np.clip(out, lo_clip - 0.25 * span_clip, hi_clip + 0.25 * span_clip)
        
        # Final safety check: replace any NaN or Inf values.
        return np.nan_to_num(out_clipped, nan=last, posinf=hi_clip, neginf=lo_clip).astype(float)

    # For n >= 4, apply a conservative damped trend model, as suggested by the template.
    # This aims for a small improvement over the strong naive baseline.

    # Calculate recent slope from the last 4 observations.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Calculate scale (standard deviation) using the last 13 points if available,
    # otherwise use the entire context. This provides a measure of recent volatility.
    # For n >= 4, context will always have at least 4 elements, so np.std is safe.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Clip the slope very conservatively to a small fraction of the scale.
    # This prevents the trend from making forecasts run away, and effectively sets
    # slope to 0 if there's no variability (scale is 0).
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
    
    # Damping factor for the trend. A value of 0.6 means the trend influence
    # diminishes quickly over the forecast horizon.
    phi = 0.6
    
    # Generate steps (1-indexed) for the prediction horizon.
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the initial forecasts: last observed value + damped cumulative trend.
    out = last + slope * np.cumsum(phi ** steps)

    # Apply robust clipping to the forecasts.
    # Determine the minimum and maximum of recent context (last 13 points or all available)
    # to define a sensible band for forecasts.
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    span = hi - lo
    
    # Clip forecasts to be within recent min/max +/- a small margin (0.25 * span).
    # If span is 0 (all recent values are identical), this clips to [lo, hi].
    out_clipped = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final safety check: replace any NaN or Inf values in the output array.
    # NaN values are replaced with 'last'. Positive/negative Infs are clipped to 'hi'/'lo'.
    return np.nan_to_num(out_clipped, nan=last, posinf=hi, neginf=lo).astype(float)