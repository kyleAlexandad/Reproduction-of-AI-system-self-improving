import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros as a safe default.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # Get the last observed value. This is the backbone for the naive forecast
    # and a starting point for any damped trend.
    last = float(context[-1])
    
    # For very short contexts (less than 4 points), revert to a pure naive forecast.
    # There isn't enough data to reliably estimate a trend.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)
    
    # Calculate a damped trend component.
    # This strategy follows the recommended conservative approach:
    # 1. Use a small window for slope estimation (last 4 points).
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))
    
    # 2. Estimate a scale (standard deviation) from recent data.
    # Use the last 13 points if available, otherwise the entire context.
    # This helps in robustly clipping the slope and forecasts.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Handle cases where scale might be zero (e.g., all recent values are identical).
    # If scale is 0, clip to 0, effectively making the slope 0.
    if scale == 0:
        slope = 0.0
    else:
        # 3. Heavily clip the slope to a small fraction of the recent scale
        # to prevent forecasts from running away from the last value.
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
    
    # Apply damping to the trend. A phi value of 0.6 means the trend effect
    # diminishes quickly over the forecast horizon.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the forecasts: last value + cumulative damped trend.
    out = last + slope * np.cumsum(phi ** steps)
    
    # Apply robust clipping to the forecasts.
    # Determine the minimum and maximum of recent context (last 13 points or full context).
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Calculate the span (range) of recent values.
    span = hi - lo

    # Clip forecasts to a band around recent observations (min/max +/- 0.25 * span).
    # This ensures forecasts stay within a historically reasonable range.
    # If span is 0 (all recent values identical), the band becomes [lo, hi].
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final safety check: replace any NaN or Inf values in the output array.
    # NaN values are replaced with 'last' (the most recent robust observation).
    # Positive Infs are replaced with 'hi' (the upper bound of recent data).
    # Negative Infs are replaced with 'lo' (the lower bound of recent data).
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)