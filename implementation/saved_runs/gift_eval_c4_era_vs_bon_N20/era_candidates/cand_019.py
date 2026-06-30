import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros as a safe default.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # Get the last observed value, which serves as the backbone for the forecast.
    last = float(context[-1])
    
    # For very short series, a pure naive (last-value) forecast is the most robust option.
    # The minimum history required for a meaningful slope calculation is 4 points.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)
    
    # Calculate a small, damped trend. This strategy is recommended to provide a
    # conservative improvement over the strong naive baseline.
    
    # Use the last 4 points to estimate a recent slope.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))
    
    # Determine a scale for the data, using the last 13 points if available,
    # otherwise the entire context. This helps in normalizing the slope.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # If the standard deviation is zero (e.g., all values are the same),
    # set scale to a small epsilon to avoid division by zero later if `scale`
    # were used in a ratio, and to ensure clipping still functions.
    # However, in this specific code, scale is used to clip slope directly,
    # so if scale is zero, slope will be clipped to [-0, 0], effectively 0, which is correct.
    # No explicit epsilon needed.
    if scale == 0:
        # If all recent values are identical, the best prediction is 'last'
        # and no trend should be applied. So, force slope to 0.
        slope = 0.0
    else:
        # Clip the slope to a small fraction of the data's scale. This ensures
        # the trend correction is tiny and prevents forecasts from running away.
        # This is a critical step for conservatism.
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
    
    # Define the damping factor (phi). A value of 0.6 means the trend contribution
    # diminishes quickly over the forecast horizon.
    phi = 0.6
    
    # Generate an array of steps for the prediction horizon.
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the point forecasts: last value plus a damped cumulative trend.
    out = last + slope * np.cumsum(phi ** steps)
    
    # Apply robust clipping to the forecasts. This keeps predictions within a
    # reasonable band derived from recent observations, plus a small margin.
    
    # Determine the minimum and maximum of recent context, similar to scale.
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Calculate the span (range) of recent values.
    span = hi - lo

    # Clip forecasts to be within recent min/max +/- 25% of the span.
    # If span is 0 (all recent values are identical), this effectively clips to `lo`.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final safety check: replace any NaN or Inf values in the output array.
    # NaN values are replaced with 'last' (the most recent robust observation).
    # Positive Infs are replaced with 'hi' (the upper bound of recent data).
    # Negative Infs are replaced with 'lo' (the lower bound of recent data).
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)