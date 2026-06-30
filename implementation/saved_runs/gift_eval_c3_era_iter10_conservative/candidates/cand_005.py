import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Handle very short contexts (less than 4 points):
    # Revert to a pure last-value naive forecast, as trend estimation is unreliable.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Estimate a recent slope from the last 4 data points.
    # This is a moving average of recent differences.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale and bounds for robust clipping.
    # Use the last 13 data points if available, otherwise use the entire context.
    if n >= 13:
        context_for_stats = context[-13:]
    else:
        context_for_stats = context

    scale = float(np.std(context_for_stats))
    lo = float(np.min(context_for_stats))
    hi = float(np.max(context_for_stats))

    # Guard against zero scale: if `scale` is 0 (e.g., all values are identical),
    # `slope` will be clipped to 0, which is the correct behavior for flat data.
    # No explicit `max(scale, epsilon)` is needed here.

    # Clip the estimated slope to a small fraction of the data's recent scale.
    # This heavily dampens the trend, preventing forecasts from running away.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damping factor (phi) to the trend over the forecast horizon.
    # A phi of 0.6 means the trend's influence decays quickly with each step.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)

    # Calculate initial forecasts: last observed value + damped cumulative trend.
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping:
    # Keep forecasts within a band around the recent minimum and maximum values.
    # This prevents extreme predictions, especially for volatile series.
    span = hi - lo
    # If span is zero (e.g., all context values are identical), the clipping
    # will constrain `out` to `lo` (which equals `hi`), ensuring a flat forecast.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Final safeguard: ensure no NaN or infinite values are returned.
    # NaNs are replaced by the 'last' observed value.
    # Positive infinity is replaced by 'hi', negative infinity by 'lo'.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)