import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    last = float(context[-1])

    # For very short contexts (fewer than 4 points), revert to a simple naive forecast
    # This prevents unstable calculations of slope or scale from very few data points.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a simple slope from the most recent 4 data points.
    # This captures short-term trend without being too sensitive to noise.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine the scale and recent value range (min/max) for robust clipping.
    # Use the last 13 points if available, otherwise use the entire context.
    # This aligns with the forecast horizon and typical weekly patterns without using full seasonality.
    if n >= 13:
        scale = float(np.std(context[-13:]))
        lo = float(np.min(context[-13:]))
        hi = float(np.max(context[-13:]))
    else:
        scale = float(np.std(context))
        lo = float(np.min(context))
        hi = float(np.max(context))
    
    # Clip the calculated slope to a small fraction of the data's scale.
    # This prevents aggressive trend extrapolation, keeping forecasts conservative.
    # If `scale` is zero (e.g., all context values are the same), `slope` will also become zero.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
    
    # Apply a damping factor to the trend.
    # A `phi` value of 0.6 ensures that the trend's influence diminishes quickly over the horizon,
    # making forecasts converge towards a stable value or the last observed value.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the forecasted values by adding the damped cumulative trend to the last observed value.
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a sensible band around recent observations.
    # This prevents forecasts from going too far beyond the historical range, even with some trend.
    # The band is defined by the min/max of recent context, extended by 25% of the span.
    span = hi - lo
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Ensure no NaN or infinite values are returned. Replace them with sensible fallbacks:
    # NaN is replaced by the last observed value.
    # Positive infinity is clipped to the recent maximum, negative infinity to the recent minimum.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)