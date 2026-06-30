import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros, as there's no past data to infer from.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Initialize `last` with the last observed value.
    last = float(context[-1])

    # Robust handling for `last` if it's NaN from a non-empty context where the last element is NaN.
    # This ensures `last` always has a valid numeric value for short contexts and as a fallback.
    if np.isnan(last):
        valid_context_values = context[~np.isnan(context)]
        if len(valid_context_values) > 0:
            last = float(valid_context_values[-1])
        else:
            # Fallback if all values in context are NaN, resulting in an undefined 'last'.
            last = 0.0

    # If context is very short (less than 4 points), we rely purely on the robust naive forecast.
    # Trend calculation requires at least 4 points for meaningful recent differences.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # --- Damped Trend Component Calculation ---
    # Use the last 4 points for slope calculation to capture very recent movement.
    recent_for_slope = context[-4:]
    slope_raw = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for clipping the raw slope.
    # Uses last 13 points if available, otherwise the entire context.
    scale_context = context[-13:] if n >= 13 else context
    
    # Calculate scale using standard deviation, only considering non-NaN values.
    # Fallback to a nominal scale (1.0) if std is zero/NaN or context has too few valid points.
    valid_scale_context = scale_context[~np.isnan(scale_context)]
    scale = float(np.std(valid_scale_context)) if len(valid_scale_context) > 1 else 1.0
    if np.isnan(scale) or scale < 1e-6:
        scale = 1.0 # Use a nominal scale to avoid division by zero or NaN issues, or make slope zero.

    slope = 0.0
    # Introduce a minimum slope ratio: only apply a trend if the raw slope is "significant"
    # relative to the recent scale. This prevents reacting to small, noisy fluctuations.
    min_slope_ratio = 0.01 
    if np.abs(slope_raw) > min_slope_ratio * scale:
        # Clip the raw slope to a conservative fraction of the recent scale, as suggested by the template.
        slope = float(np.clip(slope_raw, -0.1 * scale, 0.1 * scale))

    # Apply a damping factor (phi) to the trend. A value of 0.6 means the trend influence
    # diminishes relatively quickly over the forecast horizon.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # The forecast starts from the `last` observed value and adds a damped trend component.
    # This directly implements the "Naive + Damped Trend" approach recommended by the template.
    out = last + slope * np.cumsum(phi ** steps)

    # --- Robust Clipping ---
    # Clip forecasts to stay within a conservative band around recent observations.
    clip_context = context[-13:] if n >= 13 else context
    
    # Determine recent min/max values from valid (non-NaN) context data.
    valid_clip_context = clip_context[~np.isnan(clip_context)]
    
    # Initialize lo/hi with `last` (already robust) as a fallback if `valid_clip_context` is empty.
    lo = last
    hi = last

    if len(valid_clip_context) > 0:
        lo = float(np.min(valid_clip_context))
        hi = float(np.max(valid_clip_context))
    
    # Ensure hi is not less than lo to prevent issues in span calculation or clipping.
    if hi < lo: hi = lo

    span = hi - lo

    # If recent data is effectively constant (span is zero or very small),
    # clip forecasts to that constant value to prevent divergence.
    if span <= 1e-6:
        out = np.clip(out, lo, hi)
    else:
        # Otherwise, clip within historical min/max +/- a tighter margin (0.15 * span).
        # This keeps forecasts within sensible data-driven boundaries, crucial for random-walk like data.
        out = np.clip(out, lo - 0.15 * span, hi + 0.15 * span)
    
    # Final robust handling of NaN/Inf values:
    # Ensure no NaN or Inf values are returned. NaNs are replaced by `last` (already robust).
    # Positive Infs are clipped to `hi`, Negative Infs to `lo`.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)