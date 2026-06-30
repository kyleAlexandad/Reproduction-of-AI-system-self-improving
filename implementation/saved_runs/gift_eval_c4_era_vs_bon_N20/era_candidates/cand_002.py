import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros as a safe fallback
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback to naive forecast for very short contexts (less than 4 points),
    # where trend calculations might be unstable or unrepresentative.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # --- Trend calculation ---
    # Use the last 4 points to calculate a recent slope. `context[-4:]` is safe since `n >= 4`.
    recent = context[-4:]
    slope = np.mean(np.diff(recent)) # Mean of the last 3 differences

    # Determine a relevant scale for the series, used to clip the slope.
    # Use the last 13 points if available, otherwise use all available context.
    scale_context = context[-13:] if n >= 13 else context
    scale = np.std(scale_context)

    # Robustness for flat or near-flat series: if standard deviation is zero or very small,
    # provide a minimal positive scale. This prevents division by zero or
    # excessively large normalized slope values if context is almost constant.
    if scale < 1e-6:
        # Base minimal scale on the absolute value of the last observation, plus a small constant
        scale = np.abs(last) * 0.1 + 1e-6

    # Clip the calculated slope to a small fraction of the recent scale.
    # This is a crucial conservative step, preventing aggressive trend extrapolation.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # --- Forecast generation with damped trend ---
    phi = 0.6  # Damping factor (0.6 <= 0.7 as recommended)
    steps = np.arange(1, prediction_length + 1) # Steps for cumulative sum (1, 2, ..., prediction_length)
    
    # Generate forecasts: last value + cumulative sum of damped trend components.
    out = last + slope * np.cumsum(phi ** steps)

    # --- Robust Clipping to recent value range ---
    # Define a range based on recent observations to prevent forecasts from straying too far.
    # Use the last 13 points if available, otherwise use all available context.
    clip_context = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_context))
    hi = float(np.max(clip_context))

    # Ensure `lo` and `hi` are finite. If `clip_context` contained NaNs, `np.min`/`np.max` could yield NaN.
    # Replace any NaNs with `last` to maintain a valid clipping range.
    lo = np.nan_to_num(lo, nan=last, posinf=last, neginf=last)
    hi = np.nan_to_num(hi, nan=last, posinf=last, neginf=last)

    span = hi - lo

    # If the context is very flat (span is zero or very small), create a small symmetrical
    # band around the `last` value for clipping. This allows for minor trend corrections
    # even in a flat series without being overly restrictive (i.e., clipping to a single point).
    if span <= 1e-6:
        # Define a margin, e.g., 5% of the absolute last value, with a minimum (e.g., 1.0).
        margin = np.abs(last) * 0.05
        if margin < 1.0: # Ensure a sensible minimum margin
            margin = 1.0
        lo_clip = last - margin
        hi_clip = last + margin
    else:
        # Otherwise, use a band slightly wider than the observed min/max range (0.25 * span).
        lo_clip = lo - 0.25 * span
        hi_clip = hi + 0.25 * span

    # Apply the clipping to the generated forecasts.
    out = np.clip(out, lo_clip, hi_clip)

    # --- Final NaN/Inf handling ---
    # Replace any remaining NaNs, positive infinities, or negative infinities in the output array.
    # NaNs are replaced by `last`. Positive infinities are replaced by `hi` (from `clip_context`),
    # and negative infinities by `lo` (from `clip_context`).
    out = np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)

    return out