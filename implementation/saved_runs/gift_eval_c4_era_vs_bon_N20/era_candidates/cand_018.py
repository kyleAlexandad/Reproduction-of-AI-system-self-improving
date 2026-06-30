import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # Get the last observed value, which serves as the backbone for any conservative forecast.
    last = float(context[-1])
    
    # Fallback to pure naive forecast for very short contexts (n < 4), 
    # as trend calculation is not reliable or meaningful for such limited history.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)
    
    # --- Damped Trend Extrapolation (as per recommended conservative template) ---
    
    # Calculate a recent slope. Use the last 4 observations for this.
    # Since n >= 4 at this point, context[-4:] will always contain 4 elements.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine the scale for clipping the slope. Use the standard deviation of recent values.
    # Use the last 13 points if available, otherwise use the entire context.
    # np.std is robust for arrays with one or more elements (returns 0 for single/constant arrays).
    recent_for_scale = context[-13:] if n >= 13 else context
    scale = float(np.std(recent_for_scale)) 
    
    # Clip the slope to a small fraction of the scale to prevent aggressive extrapolation.
    # This ensures the trend contribution is minor and safe.
    # If scale is 0 (i.e., the recent series is constant), max_slope_change becomes 0,
    # correctly forcing the slope to 0 and effectively reverting to a naive forecast.
    max_slope_change = 0.1 * scale
    slope = float(np.clip(slope, -max_slope_change, max_slope_change))
    
    # Apply a damping factor (phi) to the trend. This makes the trend effect diminish 
    # significantly over the prediction horizon, reflecting the random walk behavior.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the initial forecast: last value plus damped cumulative trend.
    out = last + slope * np.cumsum(phi ** steps)
    
    # --- Robust Clipping ---
    
    # Determine the minimum and maximum of recent context for robust clipping.
    # Use the last 13 points if available, otherwise use the entire context.
    # np.min/np.max are robust for arrays with one or more elements.
    recent_for_bounds = context[-13:] if n >= 13 else context
    lo = float(np.min(recent_for_bounds))
    hi = float(np.max(recent_for_bounds))
    
    # Calculate the span (range) of recent values.
    span = hi - lo
    
    # Apply robust clipping: keep forecasts within recent min/max +/- a small margin (0.25 * span).
    # If span is 0 (e.g., all recent values are identical), this clips to `lo` (which equals `hi`),
    # effectively clamping the forecast to the last value.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final safety check: replace any NaN or Inf values in the output array.
    # NaN values are replaced with 'last' (the most recent robust observation).
    # Positive Infs are replaced with 'hi' (the upper bound of recent data).
    # Negative Infs are replaced with 'lo' (the lower bound of recent data).
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)