import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # Get the last observed value, which serves as the backbone for the forecast.
    last = float(context[-1])
    
    # For very short series, fall back to pure naive (last-value) forecast.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)
    
    # Calculate a small, damped trend based on recent observations.
    # Use the last 4 points for slope calculation.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope. Use the last 13 points or all available data.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Handle cases where scale might be zero (e.g., all recent values are identical).
    # If scale is zero, the slope clipping will effectively make slope zero, leading to naive forecast.
    if scale == 0:
        slope = 0.0
    else:
        # Clip the slope to a small fraction of the recent scale to ensure conservative trend.
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
    
    # Define a damping factor for the trend. Phi <= 0.7 is recommended for heavy damping.
    phi = 0.6
    
    # Generate steps for prediction horizon.
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate forecasts: last value + damped cumulative trend.
    out = last + slope * np.cumsum(phi ** steps)
    
    # Apply robust clipping to the forecasts.
    # Determine the minimum and maximum of recent context for clipping bounds.
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Calculate the span (range) of recent values.
    span = hi - lo

    # Clip forecasts within a band around recent observations (min/max +/- a small margin).
    # If span is 0 (e.g., all recent values are identical), this clips to `lo` (which equals `hi`).
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final safety check: replace any NaN or Inf values in the output array.
    # NaN values are replaced with 'last' (the most recent robust observation).
    # Positive Infs are replaced with 'hi' (the upper bound of recent data).
    # Negative Infs are replaced with 'lo' (the lower bound of recent data).
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)