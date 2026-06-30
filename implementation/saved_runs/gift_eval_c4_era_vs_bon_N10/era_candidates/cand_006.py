import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard Requirement 8: Handle short context arrays robustly.
    # If context is empty, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Get the last observed value, which serves as the backbone/anchor for forecasts.
    last = float(context[-1])

    # For very short context (less than 4 points needed for reliable slope calculation),
    # fallback to the pure naive (last-value) forecast. This is conservative and safe.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # --- Conservative Damped Trend Calculation (for n >= 4) ---

    # Calculate a recent slope from the last 4 observations.
    # INDEXING SAFETY: context[-4:] is safe because n >= 4.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a relevant scale for the series, using up to the last 13 observations.
    # This handles series shorter than 13 while still having enough data (n >= 4).
    # `context[-min(n, 13):]` safely extracts the relevant window.
    recent_for_scale_and_bounds = context[-min(n, 13):]

    # Calculate standard deviation as a measure of recent scale.
    # If all recent values are identical, std will be 0. This is handled gracefully.
    scale = float(np.std(recent_for_scale_and_bounds))

    # Conservatively clip the calculated slope to a small fraction of the series' scale.
    # This prevents aggressive trend extrapolation, keeping forecasts close to the last value.
    # If scale is 0, the clip range is [0, 0], effectively setting clipped_slope to 0.
    clipped_slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply the damped trend to generate forecasts.
    # `phi` is a damping factor (0.6 makes the trend impact diminish quickly).
    # `np.cumsum(phi ** steps)` weights recent steps more, then gradually less.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    out = last + clipped_slope * np.cumsum(phi ** steps)

    # --- Robust Clipping to a Band Around Recent Observations ---

    # Get min/max of the recent observations to define a plausible forecast range.
    lo = float(np.min(recent_for_scale_and_bounds))
    hi = float(np.max(recent_for_scale_and_bounds))
    span = hi - lo

    # Clip forecasts to be within a band around recent observations.
    # This prevents forecasts from diverging too much.
    # If `span` is 0 (all recent values are identical), clip to that single value.
    if span == 0:
        out = np.clip(out, lo, hi)
    else:
        # Extend the range by a small margin (25% of the span) to allow for some change.
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Hard Requirement 4: Never output NaN or inf.
    # Replace any potential NaN/inf values with safe defaults (last value or bounds).
    # This acts as a final safeguard.
    out = np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)

    # Hard Requirement 3: Output length must be EXACTLY prediction_length.
    # The `out` array is constructed to have the correct length.
    return out