import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Handle very short context arrays (e.g., less than 4 points for slope calculation):
    # fall back to naive (last value) forecast
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a trend component using the last few points
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Calculate a scale (standard deviation) from recent history to normalize the slope
    # Use last 13 points if available, otherwise the entire context
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))

    # Ensure scale is not zero to prevent division by zero or issues with clip range.
    # If scale is zero, it means the series is flat, so the slope should effectively be zero.
    # The clip below handles this correctly by setting bounds to [0, 0].
    # So, no explicit `if scale == 0: scale = 1.0` is needed here.

    # Clip the slope to a small fraction of the scale. This is crucial for conservatism,
    # preventing aggressive or runaway extrapolation.
    # The problem states "tiny correction only".
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damped trend. `phi=0.6` is a strong damping factor, ensuring the trend
    # contribution diminishes quickly over the forecast horizon.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    # The cumulative sum of phi ** steps applies the trend for each step
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a reasonable band around recent observations.
    # Define the band based on the min/max of recent data (last 13 points or full context).
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    span = hi - lo

    # Extend the band slightly (0.25 * span) to allow for some movement beyond the min/max.
    # If span is 0 (flat series), the clipping range becomes [lo, hi], which is correct.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Final safety measure: replace any NaN/inf values.
    # NaN is replaced by the `last` observed value.
    # Positive infinity is replaced by `hi`, negative infinity by `lo`.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)