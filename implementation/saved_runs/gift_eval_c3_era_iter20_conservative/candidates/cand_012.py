import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Robust determination of a fallback value for NaNs.
    # This will be used if 'last' is NaN or if the entire context is NaN.
    nan_fallback_value = 0.0
    if n > 0:
        valid_context_values = context[~np.isnan(context)]
        if len(valid_context_values) > 0:
            nan_fallback_value = float(valid_context_values[-1])
        # If all context values are NaN, nan_fallback_value remains 0.0

    # Handle empty context: return zeros, as there's no past data to infer from.
    if n == 0:
        return np.full(prediction_length, nan_fallback_value, dtype=float)

    last = float(context[-1])
    # If the very last value is NaN, try to get the last valid value from the context.
    if np.isnan(last):
        last = nan_fallback_value # Use the robust fallback calculated above

    # If context is very short (less than 4 points), we rely purely on the naive forecast
    # using the last (valid) value. Trend calculation requires at least 4 points for meaningful recent differences.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # --- Damped Trend Component Calculation ---
    # Use the last 4 points for slope calculation to capture very recent movement.
    recent_for_slope = context[-4:]
    # Calculate raw slope, ignoring NaNs in recent_for_slope.
    diffs = np.diff(recent_for_slope)
    valid_diffs = diffs[~np.isnan(diffs)]
    
    slope_raw = 0.0
    if len(valid_diffs) > 0:
        slope_raw = float(np.mean(valid_diffs))

    # Determine a scale for clipping the raw slope.
    # Uses last 13 points if available, otherwise the entire context.
    scale_context = context[-13:] if n >= 13 else context
    
    # Calculate scale using standard deviation of valid values, falling back to a default if std is zero/NaN.
    valid_scale_context = scale_context[~np.isnan(scale_context)]
    scale = 1.0 # Default nominal scale
    if len(valid_scale_context) > 1: # Need at least 2 points to calculate std
        std_val = float(np.std(valid_scale_context))
        if std_val > 1e-6:
            scale = std_val
    
    # Clip the raw slope to a conservative fraction of the recent scale, as per template.
    # This ensures the trend correction is always small relative to the data's variability.
    slope = float(np.clip(slope_raw, -0.1 * scale, 0.1 * scale))

    # Apply a damping factor to the trend. The template uses 0.6.
    # This ensures the trend's influence diminishes over the prediction horizon.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # The forecast is directly the last value plus the damped trend component.
    # This approach implies the trend directly adjusts the naive forecast,
    # rather than being blended with a tiny weight as in the parent.
    out = last + slope * np.cumsum(phi ** steps)

    # --- Robust Clipping ---
    # Clip forecasts to stay within a conservative band around recent observations.
    # This prevents extreme values and ensures stability, crucial for random-walk data.
    clip_context = context[-13:] if n >= 13 else context
    
    # Get min/max from valid values in the clipping context.
    valid_clip_context = clip_context[~np.isnan(clip_context)]
    
    lo = last
    hi = last
    if len(valid_clip_context) > 0:
        lo = float(np.min(valid_clip_context))
        hi = float(np.max(valid_clip_context))
    
    # Ensure hi is not less than lo after possibly complex calculations or NaNs.
    if hi < lo: hi = lo

    span = hi - lo

    # If recent data is effectively constant (span is zero or very small), clip to that constant value.
    if span <= 1e-6:
        final_forecast = np.clip(out, lo, hi)
    else:
        # Otherwise, clip within historical min/max +/- a margin (template uses 0.25*span).
        # This keeps forecasts within sensible data-driven boundaries.
        final_forecast = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final robust handling of NaN/Inf values:
    # Ensure no NaN or Inf values are returned. NaNs are replaced by the last valid value.
    # Infs are clipped to the recent min/max (lo/hi).
    return np.nan_to_num(final_forecast, nan=nan_fallback_value, posinf=hi, neginf=lo).astype(float)