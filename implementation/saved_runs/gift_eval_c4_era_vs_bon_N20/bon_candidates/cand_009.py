import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback to Naive for very short series (less than 4 points to compute a meaningful recent slope)
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate recent slope (trend) using the last 4 points
    # This is a very short-term trend, keeping it conservative as advised.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Calculate scale for robust clipping of the slope
    # Use the last 13 points if available, otherwise the full context
    scale_window = context[-13:] if n >= 13 else context
    scale = float(np.std(scale_window))

    # Handle case where standard deviation is zero (all values in window are identical)
    if scale == 0:
        # Use a small non-zero value to prevent division by zero or zero-range clipping.
        # This ensures the slope can still be slightly non-zero if 'last' is not zero.
        scale = np.abs(last) * 0.05 + 1e-6

    # Clip the slope to a small fraction of the scale to prevent aggressive trend extrapolation
    # This makes the trend correction very tiny, as recommended.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Damping factor for the trend, heavily damp it as recommended (e.g., phi <= 0.7)
    phi = 0.6

    # Generate the forecast steps
    # The trend effect is accumulated but decays with `phi`.
    steps = np.arange(1, prediction_length + 1)
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping to keep forecasts within a band around recent observations
    # Use the last 13 points if available, otherwise the full context for min/max
    clip_window = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_window))
    hi = float(np.max(clip_window))
    span = hi - lo

    # Handle case where min and max are the same (all values in window are identical)
    if span == 0:
        # Create a small non-degenerate clipping range around `last` value.
        # This prevents `lo - 0.25 * span` and `hi + 0.25 * span` from being degenerate if `span` is 0.
        # Ensures that even if values are identical, there's a tiny band for forecasts to exist.
        margin = np.abs(last) * 0.05 + 1e-6
        lo = last - margin
        hi = last + margin
        # Re-calculate span for clarity, though not strictly needed for clipping after adjustment.
        # span = hi - lo

    # Apply clipping to the forecast output
    # Forecasts are clipped within recent min/max +/- 25% of the span.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Handle any potential NaN or Inf values, replacing them with safe defaults
    # nan: replaced with the last observed value
    # posinf: replaced with the upper bound of the clipping range
    # neginf: replaced with the lower bound of the clipping range
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)