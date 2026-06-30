import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros as a safe fallback
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # Get the last observed value, which is the backbone of the naive forecast.
    last = float(context[-1])
    
    # For very short series, fall back to a pure naive (last-value) forecast.
    # This prevents unreliable trend calculations from insufficient data.
    if n < 4:
        out = np.full(prediction_length, last, dtype=float)
    else:
        # For longer series, apply a small, damped trend correction.
        # This strategy aligns with the "GOOD CONSERVATIVE TEMPLATE" provided in the prompt,
        # which aims for a small, safe improvement over the strong naive baseline.
        
        # Calculate slope from the most recent 4 points.
        recent = context[-4:]
        slope = float(np.mean(np.diff(recent)))
        
        # Determine a scale (standard deviation) for capping the slope.
        # Use the last 13 points if available, otherwise use the entire context.
        scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
        
        # Clip the calculated slope to a small fraction of the scale (0.1)
        # to ensure it's a "tiny correction only" and prevents aggressive extrapolation.
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
        
        # Define the damping factor (phi) for the trend. The template suggests 0.6.
        phi = 0.6
        
        # Generate steps for the prediction horizon (1 to prediction_length).
        steps = np.arange(1, prediction_length + 1)
        
        # Calculate the forecast: last value plus a cumulatively damped trend component.
        # The trend contribution diminishes with each step due to `phi ** steps`.
        out = last + slope * np.cumsum(phi ** steps)

    # Apply robust clipping to the forecasts. This ensures forecasts stay within
    # a sensible range around recent observations, preventing them from diverging wildly.
    
    # Determine the minimum and maximum of recent context values for setting clipping bounds.
    # Use the last 13 points if available, otherwise use the entire context.
    # For very short contexts (e.g., n=1, 2, 3), this will correctly use the few available points.
    recent_for_bounds = context if n < 13 else context[-13:]
    lo = float(np.min(recent_for_bounds))
    hi = float(np.max(recent_for_bounds))
    
    # Calculate the span (range) of these recent values.
    span = hi - lo

    # Clip forecasts using bounds that are the recent min/max +/- a small margin (0.25 * span).
    # This is a robust way to keep forecasts anchored to recent historical patterns.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final safety check: replace any NaN or Inf values in the output array.
    # NaN values are replaced with 'last' (the most recent robust observation).
    # Positive Infs are replaced with 'hi' (the upper bound of recent historical data).
    # Negative Infs are replaced with 'lo' (the lower bound of recent historical data).
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)