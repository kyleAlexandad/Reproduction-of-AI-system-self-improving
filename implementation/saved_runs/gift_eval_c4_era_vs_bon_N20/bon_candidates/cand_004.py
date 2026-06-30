import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Handle very short context (less than 4 points): revert to naive last-value
    # This prevents issues with diff/mean for slope calculation or std for scale.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend component
    # Use the last 4 observations to estimate a recent slope
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope and the final forecasts.
    # Use last 13 values if available, otherwise all available context.
    # This helps in adapting to the series' recent volatility.
    if n >= 13:
        scale_context = context[-13:]
    else:
        scale_context = context
    
    # Add a small epsilon to standard deviation to prevent division by zero or very small values
    # if the series is constant or has very low variance.
    scale = float(np.std(scale_context))
    
    # Clip the slope to a small fraction of the scale to ensure it's a tiny correction.
    # This prevents aggressive trend extrapolation, aligning with random walk behavior.
    # If scale is zero (e.g., all recent values are identical), slope will be clipped to zero.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damping factor (phi <= 0.7 recommended)
    # A value of 0.6 ensures the trend effect diminishes quickly over the forecast horizon.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate forecasts as last value + damped cumulative slope
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations.
    # This prevents forecasts from running away, especially for noisy series.
    # Use the same context window as for scale.
    lo = float(np.min(scale_context))
    hi = float(np.max(scale_context))
    span = hi - lo
    
    # Define the clipping band as recent min/max +/- a small margin (25% of the span).
    # If span is zero (all values in scale_context are same), lo and hi will be equal,
    # and the clipping will effectively force `out` to `lo` (which is `last`).
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Final safety: replace any potential NaN/inf with sane values.
    # NaN values are replaced by the last observed value.
    # Positive infinity by the recent maximum, negative infinity by the recent minimum.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)