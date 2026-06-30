import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # Get the last observed value. This will be the backbone of our forecast.
    last = float(context[-1])
    
    # For very short series, fall back to a pure naive (last-value) forecast.
    # The slope calculation needs at least 2 points for diff, and we use 4 points for 'recent'.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)
    
    # Calculate a damped trend component.
    # Use the last 4 observations to compute a recent slope.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))
    
    # Determine a scale factor from recent data (last 13 points if available, otherwise all context).
    # This helps to clip the slope to a reasonable magnitude.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Ensure scale is not zero to prevent division by zero or overly aggressive clipping if all values are identical.
    # If scale is zero, it means context values are constant, so slope should be zero.
    if scale == 0:
        clipped_slope = 0.0
    else:
        # Clip the slope to a small fraction (10%) of the recent scale.
        # This makes the trend correction very conservative.
        clipped_slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
        
    # Apply a strong damping factor (phi <= 0.7 as recommended) to the trend.
    # A value of 0.6 means the trend effect diminishes quickly over the forecast horizon.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the forecast: last value + damped cumulative trend.
    out = last + clipped_slope * np.cumsum(phi ** steps)
    
    # Apply robust clipping to the forecasts to keep them within a reasonable historical band.
    # Determine the minimum and maximum of recent context (last 13 points or all).
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Calculate the span (range) of recent values.
    span = hi - lo

    # Clip forecasts to be within recent min/max +/- a small margin (0.25 * span).
    # If span is 0 (e.g., all recent values are identical), this effectively clips to `lo`.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final safety check: replace any NaN or Inf values in the output array.
    # NaN values are replaced with 'last' (the most recent robust observation).
    # Positive Infs are replaced with 'hi' (the upper bound of recent data).
    # Negative Infs are replaced with 'lo' (the lower bound of recent data).
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)