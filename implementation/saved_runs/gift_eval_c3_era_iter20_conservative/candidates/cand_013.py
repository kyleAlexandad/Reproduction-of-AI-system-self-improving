import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Determine a robust replacement value for NaN `last` in short contexts or generally.
    # This value will be used as a fallback for NaNs and when context is too short.
    nan_replacement_value = 0.0
    if n > 0:
        valid_context_values = context[~np.isnan(context)]
        if len(valid_context_values) > 0:
            nan_replacement_value = float(valid_context_values[-1])
        # If all context values are NaN, nan_replacement_value remains 0.0

    # Handle empty context: return zeros, or nan_replacement_value if it's non-zero
    # but for n=0, 0.0 is the safest default as there's no data to infer from.
    if n == 0:
        return np.full(prediction_length, 0.0, dtype=float)

    last = float(context[-1])
    # If the last value itself is NaN, use the robust replacement.
    if np.isnan(last):
        last = nan_replacement_value

    # If context is very short (less than 4 points), we rely purely on the naive forecast.
    # Trend calculation requires at least 4 points for meaningful recent differences.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # --- Damped Trend Component Calculation ---
    # Use the last 4 points for slope calculation to capture very recent movement.
    recent_for_slope = context[-4:]
    
    # Handle potential NaNs in recent_for_slope for diff calculation
    valid_recent_for_slope = recent_for_slope[~np.isnan(recent_for_slope)]
    
    slope_raw = 0.0
    if len(valid_recent_for_slope) >= 2: # Need at least 2 points to calculate differences
        slope_raw = float(np.mean(np.diff(valid_recent_for_slope)))

    # Determine a scale for clipping the raw slope.
    # Uses last 13 points if available, otherwise the entire context.
    scale_context = context[-13:] if n >= 13 else context
    
    # Filter NaNs for std calculation.
    valid_scale_context = scale_context[~np.isnan(scale_context)]
    
    scale = 1.0 # Default scale to avoid division by zero or NaN issues
    if len(valid_scale_context) > 1: # Need at least 2 points for std
        std_val = float(np.std(valid_scale_context))
        if not np.isnan(std_val) and std_val >= 1e-6:
            scale = std_val

    slope = 0.0
    # Introduce a minimum slope ratio: only apply a trend if the raw slope is "significant"
    # relative to the recent scale, preventing reactions to small, noisy fluctuations.
    min_slope_ratio = 0.01 # Raw slope must be at least 1% of scale to activate trend
    if np.abs(slope_raw) > min_slope_ratio * scale:
        # Clip the raw slope to a fraction of the recent scale.
        # Adjusted from 0.05 (parent) to 0.075 to allow slightly more trend influence.
        slope = float(np.clip(slope_raw, -0.075 * scale, 0.075 * scale))

    # Apply a damping factor to the trend. Its influence diminishes rapidly over the horizon.
    # Adjusted from 0.4 (parent) to 0.5 for slightly less aggressive damping.
    phi = 0.5
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the incremental damped trend component for each step.
    damped_trend_component = slope * np.cumsum(phi ** steps)

    # --- Ensemble Blending ---
    # Blend the naive forecast (simple repetition of the last value) with the damped trend.
    # Given the random walk behavior, the naive component is still dominant.
    # Adjusted from 0.9999 (parent) to 0.95, allowing the damped trend to have 5% influence.
    weight_naive = 0.95
    weight_damped_trend_influence = 1.0 - weight_naive # This will be 0.05

    # The blended forecast is primarily the last value, with a minimal, damped trend correction.
    blended_forecast = last + weight_damped_trend_influence * damped_trend_component

    # --- Robust Clipping ---
    # Clip forecasts to stay within a conservative band around recent observations.
    clip_context = context[-13:] if n >= 13 else context
    
    # Handle cases where clip_context might be empty or all NaNs, to get valid min/max.
    # Filter NaNs before min/max.
    valid_clip_context = clip_context[~np.isnan(clip_context)]
    
    lo = last # Default to last value
    hi = last # Default to last value

    if len(valid_clip_context) > 0:
        lo = float(np.min(valid_clip_context))
        hi = float(np.max(valid_clip_context))
    
    # Ensure lo and hi are valid numbers. If they are NaN or Inf, use the last value.
    if np.isnan(lo) or np.isinf(lo): lo = last
    if np.isnan(hi) or np.isinf(hi): hi = last

    # Ensure hi is not less than lo.
    if hi < lo: hi = lo

    span = hi - lo

    final_forecast = np.copy(blended_forecast) # Work on a copy

    # If recent data is effectively constant (span is zero or very small), clip to that constant value.
    if span <= 1e-6:
        final_forecast = np.clip(final_forecast, lo, hi)
    else:
        # Otherwise, clip within historical min/max +/- a tighter margin (0.15*span from parent).
        # This margin is tighter than the template's 0.25 and is kept for conservatism.
        final_forecast = np.clip(final_forecast, lo - 0.15 * span, hi + 0.15 * span)
    
    # Final robust handling of NaN/Inf values:
    # Ensure no NaN or Inf values are returned. NaNs are replaced by the last valid value (robustly determined).
    # Infs are clipped to the recent min/max (lo/hi).
    return np.nan_to_num(final_forecast, nan=last, posinf=hi, neginf=lo).astype(float)