import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # Get the last observed value, which serves as the backbone for naive forecast
    last = float(context[-1])
    
    # Handle very short contexts (less than 4 points):
    # Fallback to pure naive (last-value) forecast, as trend calculation might be unstable
    if n < 4:
        return np.full(prediction_length, last, dtype=float)
    
    # Calculate a short-term trend (slope)
    # Use the last 4 points for slope calculation to capture very recent changes
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent))) # Mean difference of last 4 points

    # Calculate a scale to normalize and clip the slope
    # Use the standard deviation of the last 13 points (or full context if shorter)
    # This helps in preventing the slope from being too aggressive relative to recent variability
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # If the series is flat (std is 0), then scale is 0. 
    # In this case, the slope will be correctly clipped to 0, making the forecast flat.
    
    # Clip the slope to a small fraction (10%) of the recent scale
    # This makes the trend correction tiny and conservative, preventing forecasts from running away
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
    
    # Damping factor for the trend
    # A value of 0.6 ensures the trend contribution diminishes quickly over the forecast horizon,
    # making the forecasts converge towards the last observed value.
    phi = 0.6
    
    # Calculate steps for the cumulative sum of the damped trend
    steps = np.arange(1, prediction_length + 1)
    
    # Generate initial forecasts: last value + damped trend component
    out = last + slope * np.cumsum(phi ** steps)
    
    # Robust clipping of forecasts within a band around recent observations
    # Determine the minimum and maximum of recent context (last 13 points or full context)
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Calculate the span (range) of recent values
    span = hi - lo

    # Apply robust clipping: keep forecasts within recent min/max +/- a small margin (0.25 * span)
    # This prevents extreme forecasts and ensures they stay within a reasonable historical range.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final safety check: replace any NaN or Inf with sane values
    # NaN values are replaced with 'last' (a robust default)
    # Positive Infs are replaced with 'hi' (the upper bound of recent data)
    # Negative Infs are replaced with 'lo' (the lower bound of recent data)
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)