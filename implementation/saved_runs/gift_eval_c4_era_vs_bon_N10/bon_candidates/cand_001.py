import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    # If the context is empty, return an array of zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # If the context is very short (less than 4 points are needed for a stable trend calculation),
    # fall back to the naive (last value) forecast.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate recent trend (slope).
    # We use the last 4 points of the context to compute the slope.
    # The `min(n, 4)` ensures we don't try to slice beyond the array length for very short `n`.
    recent_for_slope = context[-min(n, 4):]
    
    slope = 0.0 # Initialize slope to 0
    # A slope can only be calculated if there are at least two points in the `recent_for_slope` array.
    if len(recent_for_slope) >= 2:
        # Calculate the mean of differences between consecutive points to get the average slope.
        slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for robustly clipping the slope.
    # We use the standard deviation of the last 13 points if available, otherwise the entire context.
    if n >= 13:
        scale = float(np.std(context[-13:]))
    else:
        scale = float(np.std(context))
    
    # Ensure the calculated scale is not effectively zero.
    # A very small scale (e.g., from constant data) could lead to overly aggressive clipping or issues.
    # If scale is near zero, provide a fallback scale (at least 1.0, or 10% of the last value).
    if scale < 1e-6: # Using a small epsilon for floating-point comparison
        scale = max(1.0, np.abs(last) * 0.1) 

    # Clip the slope to a small fraction (10%) of the calculated scale.
    # This is a critical step for conservatism, preventing aggressive trend extrapolation.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Damping factor for the trend contribution.
    # A value of 0.6 means the trend's influence diminishes quickly over the forecast horizon.
    phi = 0.6

    # Calculate the forecast.
    # The forecast is built upon the last observed value, with a damped trend component.
    # np.cumsum(phi ** steps) provides the cumulative sum of the damped trend contribution for each step.
    steps = np.arange(1, prediction_length + 1)
    out = last + slope * np.cumsum(phi ** steps)

    # Robust Clipping: Keep forecasts within a sensible band around recent observations.
    # Determine the minimum and maximum values from the last 13 points, or the entire context if shorter.
    if n >= 13:
        lo = float(np.min(context[-13:]))
        hi = float(np.max(context[-13:]))
    else:
        lo = float(np.min(context))
        hi = float(np.max(context))

    span = hi - lo
    
    # Handle the edge case where recent values are all identical (span is effectively zero).
    # If span is zero, the clipping range would be a single point, which is too restrictive for forecasting.
    if span < 1e-6: # Using a small epsilon for floating-point comparison
        # Create a small, dynamic margin to define a non-zero clipping band around 'last'.
        # This margin is at least 0.1 and also proportional to `last` if `last` is large.
        margin = max(0.01 * np.abs(last), 0.1) 
        lo = last - margin
        hi = last + margin

    # Clip the output forecasts.
    # Forecasts are constrained within a band defined by the recent min/max, extended by 25% of the span on each side.
    # This acts as a safety net against runaway predictions.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Hard requirement 4: Never output NaN or inf.
    # Use np.nan_to_num to replace any NaNs with `last` and positive/negative Infs with `hi`/`lo`.
    # This ensures the output array is always finite and numerically stable.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)