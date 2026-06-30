import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    last = float(context[-1])
    
    # Handle very short context arrays by falling back to naive (last value) forecast.
    # This ensures safe indexing for trend calculation and provides a robust baseline.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)
    
    # Calculate a small, damped trend. This strategy is recommended to potentially
    # achieve a slight improvement over the strong naive baseline.
    
    # Use the last 4 points to calculate a recent slope.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))
    
    # Determine a scale for clipping the slope. Use the standard deviation of
    # the last 13 points if available, otherwise use the entire context.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Ensure scale is not zero to prevent division by zero if all values are identical.
    # If scale is zero, it means no variability, so a zero slope is appropriate.
    if scale == 0:
        slope = 0.0
    else:
        # Clip the slope to a small fraction of the recent scale to ensure
        # the trend correction is very conservative and does not cause forecasts to run away.
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
    
    # Apply a damping factor (phi) to the trend. A value of 0.6 heavily
    # dampens the trend, making forecasts quickly converge back to naive for longer horizons.
    phi = 0.6
    
    # Calculate steps for the forecast horizon.
    steps = np.arange(1, prediction_length + 1)
    
    # Generate forecasts by adding the damped trend to the last observed value.
    out = last + slope * np.cumsum(phi ** steps)
    
    # Apply robust clipping to the forecasts. This helps to ensure that forecasts
    # stay within a reasonable band derived from recent observations.
    
    # Determine the minimum and maximum of recent context for clipping.
    # Use the last 13 points if available, otherwise use the entire context.
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Calculate the span (range) of recent values.
    span = hi - lo

    # Clip forecasts to a band around recent min/max (+/- 0.25 * span).
    # This prevents extreme forecasts if the damped trend was slightly off.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final safety check: replace any NaN or Inf values in the output array.
    # NaN values are replaced with 'last' (the most recent robust observation).
    # Positive Infs are replaced with 'hi' (the upper bound of recent data).
    # Negative Infs are replaced with 'lo' (the lower bound of recent data).
    # This ensures no invalid values are returned.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)