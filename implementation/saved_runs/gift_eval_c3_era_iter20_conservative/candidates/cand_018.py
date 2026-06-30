import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short contexts:
    # If no data, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # Get the last observed value, which is the anchor for naive forecast.
    last = float(context[-1])

    # If context is very short (e.g., less than 4 points for trend calculation),
    # fall back to simple naive (last-value) forecast.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend component.
    # Use the last 4 points to estimate a recent slope.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for the data (e.g., standard deviation of recent values).
    # This helps in clipping the slope and forecasts safely.
    # Use last 13 points if available, otherwise use all available context.
    scale_context = context[-13:] if n >= 13 else context
    # Handle cases where scale_context might have zero std (e.g., all values are same)
    # to prevent division by zero or issues with relative clipping.
    scale = float(np.std(scale_context))
    if scale == 0:
        scale = np.mean(np.abs(scale_context)) # Use mean absolute value as scale if std is zero
        if scale == 0: # If all values are zero, scale remains zero.
            scale = 1.0 # Use a small default scale to avoid issues with 0.1 * scale clipping.

    # Clip the estimated slope to a small fraction of the scale.
    # This ensures the trend correction is very conservative and does not cause forecasts to run away.
    # The prompt explicitly warns against aggressive trend extrapolation.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a strong damping factor to the trend.
    # A phi of 0.6 means the trend effect quickly diminishes over the prediction horizon.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the forecast: last value + damped trend component.
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations.
    # This prevents forecasts from becoming unrealistic (too high or too low).
    # Use min/max of the recent context (last 13 points or all available).
    min_val_context = context[-13:] if n >= 13 else context
    lo = float(np.min(min_val_context))
    hi = float(np.max(min_val_context))
    span = hi - lo
    
    # If span is zero (all values in context are the same), adjust for clipping.
    if span == 0:
        # If all context values are the same, forecast should be that value.
        # Clipping to [lo, hi] which are both 'last' would work, but adding a margin
        # relative to 0.25 * span (which is 0) means no margin.
        # To be safe, if all values are same, we ensure out is exactly 'last'.
        # The slope would have been clipped to zero if std was zero, so 'out' would already be 'last'.
        # This check primarily guards against floating point edge cases or very small span.
        out = np.clip(out, lo, hi)
    else:
        # Clip forecasts to a band around the recent min/max, with a small margin (25% of the span).
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Ensure no NaN or infinite values in the output, replacing them safely.
    # NaNs replaced by 'last', positive infinity by 'hi', negative infinity by 'lo'.
    out = np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)
    
    return out