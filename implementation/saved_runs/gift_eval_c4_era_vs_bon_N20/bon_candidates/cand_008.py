import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short contexts:
    # If no data, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # For contexts too short to reliably calculate trend (e.g., less than 4 points),
    # fallback to the simple last-value naive forecast.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a recent slope for a tiny damped trend correction.
    # Use the last 4 points for slope to capture very recent movement.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Calculate a scale (standard deviation) based on recent history
    # to bound the trend correction. Use up to the last 13 points.
    scale_context = context[-13:] if n >= 13 else context
    scale = float(np.std(scale_context))

    # If the scale is zero (e.g., recent values are all identical),
    # the slope should effectively be zero, which is handled by the clip below.
    # No explicit adjustment needed for scale=0, as max_slope_change will be 0.

    # Clip the slope to a small fraction of the recent scale (e.g., 10%).
    # This prevents aggressive trend extrapolation, keeping forecasts close to naive.
    max_slope_change = 0.1 * scale
    slope = float(np.clip(slope, -max_slope_change, max_slope_change))

    # Apply a damping factor (phi <= 0.7 recommended) to the trend.
    # This ensures the trend quickly diminishes over the forecast horizon.
    phi = 0.6

    # Generate steps for the forecast horizon (1, 2, ..., prediction_length).
    steps = np.arange(1, prediction_length + 1)

    # Calculate the raw forecasts: last value + cumulative damped trend contribution.
    out = last + slope * np.cumsum(phi ** steps)

    # Apply robust clipping to keep forecasts within a sensible band around recent observations.
    # Define the band based on the min/max of the recent context (last 13 points or all available).
    clip_context = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_context))
    hi = float(np.max(clip_context))
    span = hi - lo

    # Clip the forecasts: values are kept within [lo - 0.25*span, hi + 0.25*span].
    # This prevents forecasts from diverging too far from the observed range.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Final safety measure: replace any NaN or infinite values.
    # NaNs are replaced by the last observed value.
    # Positive infinity is replaced by the upper clip bound (hi).
    # Negative infinity is replaced by the lower clip bound (lo).
    out = np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)

    return out