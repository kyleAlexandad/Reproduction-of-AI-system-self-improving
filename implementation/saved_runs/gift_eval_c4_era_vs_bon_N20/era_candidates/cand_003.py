import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last_value = float(context[-1])

    # Handle very short context arrays (fewer than 4 points): fall back to naive (last value repeat).
    # This prevents issues with subsequent slicing and statistical calculations on too few data points.
    if n < 4:
        return np.full(prediction_length, last_value, dtype=float)

    # --- Damped Trend Calculation ---
    
    # Calculate slope based on the last 4 observations.
    # `context[-4:]` is safe because `n >= 4` is guaranteed at this point.
    recent_for_slope = context[-4:]
    slope_diffs = np.diff(recent_for_slope)
    
    # Calculate mean slope. `slope_diffs` will have 3 elements here.
    slope = float(np.mean(slope_diffs))

    # Determine a scale for clipping the slope and the final forecast values.
    # Use the last 13 observations if available, otherwise use the entire context.
    # `min(n, 13)` ensures we don't try to slice beyond the array length.
    context_for_scale_and_clip = context[-min(n, 13):]
    
    # Calculate standard deviation for scaling. If std is 0 (all values are the same),
    # the `max_slope_change` will be 0, effectively neutralizing the trend, which is
    # a desired conservative behavior for constant series.
    scale = float(np.std(context_for_scale_and_clip))
    
    # Clip the calculated slope to be a small fraction of the data's scale.
    # This prevents the trend from making forecasts run away too quickly.
    # If `scale` is 0, `max_slope_change` will be 0, forcing `slope` to 0.
    max_slope_change = 0.1 * scale
    slope = np.clip(slope, -max_slope_change, max_slope_change)

    # Apply a damping factor to the trend over the prediction horizon.
    # `phi = 0.6` is a conservative damping factor (0 < phi < 1).
    steps = np.arange(1, prediction_length + 1)
    
    # The `phi ** steps` term creates a geometrically decaying sequence (phi, phi^2, ...).
    # `np.cumsum` integrates this, so the trend effect accumulates but diminishes over time.
    damped_trend_correction = slope * np.cumsum(phi ** steps)
    
    # Initial forecast: last observed value plus the cumulative damped trend correction.
    out = last_value + damped_trend_correction

    # --- Robust Clipping ---
    
    # Determine the lower and upper bounds for clipping based on recent observations.
    # These are derived from the same `context_for_scale_and_clip` used for `scale`.
    lo = float(np.min(context_for_scale_and_clip))
    hi = float(np.max(context_for_scale_and_clip))
    
    # Calculate the span of recent observations.
    span = hi - lo
    
    # Clip the forecasts to stay within a band around recent min/max.
    # If `span` is 0 (i.e., `lo == hi`), `0.25 * span` is 0, so forecasts are clipped
    # strictly between `lo` and `hi` (which are the same value). This means forecasts
    # for a constant series will be constant, which is correct.
    clipped_out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Handle any remaining NaN or Inf values by replacing them with sensible defaults.
    # `nan=last_value`: If a value becomes NaN, revert to the most conservative forecast.
    # `posinf=hi`, `neginf=lo`: If a value becomes infinite, clip to the historical bounds.
    final_forecast = np.nan_to_num(clipped_out, nan=last_value, posinf=hi, neginf=lo).astype(float)
    
    return final_forecast