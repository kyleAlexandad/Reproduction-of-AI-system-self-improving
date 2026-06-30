import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Get the last observed value, which serves as the backbone for naive forecast.
    last = float(context[-1])

    # For very short series, fall back to pure naive (last value) forecast.
    # This ensures indexing safety and reasonable behavior for minimal data.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend for a small, conservative correction.
    # This follows the recommended conservative strategy.
    
    # Use the last 4 points to estimate the recent slope.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope. Use last 13 points or full context.
    # This prevents the trend from becoming too aggressive.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Clip the slope to a small fraction (10%) of the recent scale.
    # This is crucial for conservatism, preventing forecasts from running away.
    # If scale is 0 (e.g., all recent values are identical), slope will be clipped to 0.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damping factor to the trend. phi <= 0.7 is recommended.
    phi = 0.6
    
    # Generate cumulative sum of damped trend for each forecast step.
    steps = np.arange(1, prediction_length + 1)
    out = last + slope * np.cumsum(phi ** steps)

    # Apply robust clipping to the forecasts.
    # This keeps forecasts within a sensible band around recent observations.
    
    # Determine the minimum and maximum of recent context.
    # Use the last 13 points if available, otherwise use the entire context.
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Calculate the span (range) of recent values.
    span = hi - lo

    # Clip forecasts to be within recent min/max +/- a small margin (0.25 * span).
    # If span is 0 (e.g., all recent values are identical), this clips to `lo`.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Final safety check: replace any NaN or Inf values in the output array.
    # NaN values are replaced with 'last' (the most recent robust observation).
    # Positive Infs are replaced with 'hi' (the upper bound of recent data).
    # Negative Infs are replaced with 'lo' (the lower bound of recent data).
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)