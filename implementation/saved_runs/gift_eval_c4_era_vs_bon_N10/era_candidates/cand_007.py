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
    # `context[-min(n, 13):]` safely extracts the relevant window.
    recent_for_scale_and_bounds = context[-min(n, 13):]

    # Calculate standard deviation as a measure of recent scale.
    scale = float(np.std(recent_for_scale_and_bounds))

    # Conservatively clip the calculated slope to a small fraction of the series' scale.
    # Add a small epsilon guard for scale=0 to ensure clipping range is well-defined.
    if scale > 1e-6:
        clipped_slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
    else:
        clipped_slope = 0.0 # If scale is zero, no meaningful trend correction

    # Apply a damping factor to the trend's influence over time.
    # `phi` is 0.6, making the trend impact diminish quickly.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the pure trend correction component, relative to the last value.
    trend_correction_component = clipped_slope * np.cumsum(phi ** steps)

    # --- Conservative Blending with Naive Forecast ---
    # The prompt recommends preferring conservative ensembles/blends where NAIVE keeps a HIGH weight.
    # The parent candidate applied the full damped trend correction. Given that the parent's MASE
    # was slightly worse than the naive baseline, we further reduce the influence of the trend
    # correction to pull the forecast closer to the strong naive baseline.
    
    # We apply a small weight to the trend correction component.
    # This is equivalent to blending `last` (with weight 1-weight_trend)
    # and `(last + trend_correction_component)` (with weight weight_trend).
    # A weight of 0.15 means a 85% weight on pure naive and 15% on the damped trend forecast.
    weight_trend_influence = 0.15

    # The effective forecast is `last` plus a heavily reduced fraction of the trend correction.
    out_unclipped = last + trend_correction_component * weight_trend_influence

    # --- Robust Clipping to a Band Around Recent Observations ---

    # Get min/max of the recent observations to define a plausible forecast range.
    lo = float(np.min(recent_for_scale_and_bounds))
    hi = float(np.max(recent_for_scale_and_bounds))
    span = hi - lo

    # Clip forecasts to be within a band around recent observations.
    # If `span` is effectively zero (e.g., all recent values are identical), clip to that single value.
    # Otherwise, extend the range by a small margin (25% of the span) to allow for some change.
    if span <= 1e-6: # Use a small epsilon for comparison to zero
        out = np.clip(out_unclipped, lo, hi)
    else:
        out = np.clip(out_unclipped, lo - 0.25 * span, hi + 0.25 * span)

    # Hard Requirement 4: Never output NaN or inf.
    # Replace any potential NaN/inf values with safe defaults (last value or bounds).
    # This acts as a final safeguard.
    out = np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)

    # Hard Requirement 3: Output length must be EXACTLY prediction_length.
    # The `out` array is constructed to have the correct length.
    return out