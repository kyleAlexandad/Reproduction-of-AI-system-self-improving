import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros, as there's no past data to infer from.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Determine a robust replacement value for NaN 'last' in any context, particularly short ones.
    # This value will also be used as a fallback if other calculations yield NaN/Inf.
    nan_replacement_value = last
    if np.isnan(last):
        valid_context_values = context[~np.isnan(context)]
        if len(valid_context_values) > 0:
            nan_replacement_value = float(valid_context_values[-1])
        else:
            nan_replacement_value = 0.0 # Fallback if all values in context are NaN

    # If context is very short (less than 4 points), we rely purely on the naive forecast.
    # Trend calculation requires at least 4 points for meaningful recent differences.
    if n < 4:
        return np.full(prediction_length, nan_replacement_value, dtype=float)

    # --- Damped Trend Component Calculation ---
    # Use the last 4 points for slope calculation to capture very recent movement.
    recent_for_slope = context[-4:]
    slope_raw = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for clipping the raw slope.
    # Uses last 13 points if available, otherwise the entire context.
    scale_context = context[-13:] if n >= 13 else context
    # Calculate scale using standard deviation, ignoring NaNs. Fallback to 1.0 if std is zero/NaN or context too short.
    valid_scale_context = scale_context[~np.isnan(scale_context)]
    scale = 1.0 # Default nominal scale
    if len(valid_scale_context) > 1: # Need at least 2 points for a meaningful standard deviation
        std_val = float(np.std(valid_scale_context))
        if not np.isnan(std_val) and std_val > 1e-6:
            scale = std_val
        
    slope = 0.0
    # Introduce a minimum slope ratio: only apply a trend if the raw slope is "significant"
    # relative to the recent scale, preventing reactions to small, noisy fluctuations.
    min_slope_ratio = 0.01 # Raw slope must be at least 1% of scale to activate trend
    if np.abs(slope_raw) > min_slope_ratio * scale:
        # Clip the raw slope to a fraction of the recent scale.
        # This is slightly less aggressive than the parent (0.05) but still conservative (0.075).
        slope = float(np.clip(slope_raw, -0.075 * scale, 0.075 * scale))

    # Apply a damping factor to the trend. Its influence diminishes rapidly over the horizon.
    # A phi of 0.5 (vs 0.4 in parent) means slightly slower damping, giving the trend a bit more presence.
    phi = 0.5
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the incremental damped trend component for each step.
    damped_trend_component = slope * np.cumsum(phi ** steps)

    # --- Ensemble Blending ---
    # Blend the naive forecast (simple repetition of the last value) with the damped trend.
    # Given the random walk behavior, the naive component still receives a very high weight,
    # but the trend component is now given a noticeable (5%) influence, rather than almost zero (0.01% in parent).
    # This aligns better with the recommendation of 0.75-0.9 naive weight.
    weight_naive = 0.95
    weight_damped_trend_influence = 1.0 - weight_naive # This is 0.05

    # The blended forecast is primarily the last value, with a small, damped trend correction.
    blended_forecast = last + weight_damped_trend_influence * damped_trend_component

    # --- Robust Clipping ---
    # Clip forecasts to stay within a conservative band around recent observations.
    clip_context = context[-13:] if n >= 13 else context
    
    # Ensure lo and hi are robustly calculated from valid numbers.
    valid_clip_context = clip_context[~np.isnan(clip_context)]
    
    # Initialize lo and hi with the determined nan_replacement_value.
    lo = nan_replacement_value
    hi = nan_replacement_value

    if len(valid_clip_context) > 0:
        lo = float(np.min(valid_clip_context))
        hi = float(np.max(valid_clip_context))

    # Ensure hi is not less than lo (e.g., if context was all same value or min/max initialization issues).
    if hi < lo:
        hi = lo

    span = hi - lo

    final_forecast = np.copy(blended_forecast) # Work on a copy

    # If recent data is effectively constant (span is zero or very small), clip to that constant value.
    if span <= 1e-6:
        final_forecast = np.clip(final_forecast, lo, hi)
    else:
        # Otherwise, clip within historical min/max +/- a moderate margin.
        # This margin is tighter than the template (0.25) but looser than the parent (0.15).
        final_forecast = np.clip(final_forecast, lo - 0.2 * span, hi + 0.2 * span)
    
    # Final robust handling of NaN/Inf values.
    # Ensure no NaN or Inf values are returned. NaNs are replaced by the robust nan_replacement_value.
    # Infs are clipped to the recent min/max (lo/hi).
    return np.nan_to_num(final_forecast, nan=nan_replacement_value, posinf=hi, neginf=lo).astype(float)