import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros, as there's no past data to infer from.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # If context is very short (less than 4 points), we rely purely on the naive forecast.
    # Trend calculation requires at least 4 points for meaningful recent differences.
    if n < 4:
        # Determine a robust replacement value for NaN `last` in short contexts.
        nan_replacement_for_short = last
        if np.isnan(last):
            valid_context_values = context[~np.isnan(context)]
            if len(valid_context_values) > 0:
                nan_replacement_for_short = float(valid_context_values[-1])
            else:
                nan_replacement_for_short = 0.0 # Fallback if all values in short context are NaN
        return np.full(prediction_length, nan_replacement_for_short, dtype=float)

    # --- Damped Trend Component Calculation ---
    # Use the last 4 points for slope calculation to capture very recent movement.
    recent_for_slope = context[-4:]
    slope_raw = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for clipping the raw slope.
    # Uses last 13 points if available, otherwise the entire context.
    scale_context = context[-13:] if n >= 13 else context
    # Calculate scale using standard deviation, falling back to a default if std is zero/NaN.
    scale = float(np.std(scale_context))
    if np.isnan(scale) or scale < 1e-6:
        scale = 1.0 # Use a nominal scale to avoid division by zero or NaN issues, or make slope zero.

    slope = 0.0
    # Introduce a minimum slope ratio: only apply a trend if the raw slope is "significant"
    # relative to the recent scale, preventing reactions to small, noisy fluctuations.
    min_slope_ratio = 0.01 # Raw slope must be at least 1% of scale to activate trend
    if np.abs(slope_raw) > min_slope_ratio * scale:
        # Clip the raw slope to an even tighter fraction of the recent scale.
        # This makes the trend component extremely conservative (0.05 vs. 0.075 in parent).
        slope = float(np.clip(slope_raw, -0.05 * scale, 0.05 * scale))

    # Apply a damping factor to the trend. Its influence diminishes rapidly over the horizon.
    # A smaller phi (0.4 vs 0.5 in parent) ensures faster damping, making the trend effect short-lived.
    phi = 0.4
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the incremental damped trend component for each step.
    damped_trend_component = slope * np.cumsum(phi ** steps)

    # --- Ensemble Blending ---
    # Blend the naive forecast (simple repetition of the last value) with the damped trend.
    # Given the random walk behavior of the data and the strength of the naive baseline,
    # the naive component is given a very high weight, making the trend influence almost negligible.
    # (0.9999 vs 0.99 in parent, meaning 0.0001 trend influence vs 0.01).
    weight_naive = 0.9999
    weight_damped_trend_influence = 1.0 - weight_naive

    # The blended forecast is primarily the last value, with a minimal, damped trend correction.
    blended_forecast = last + weight_damped_trend_influence * damped_trend_component

    # --- Robust Clipping ---
    # Clip forecasts to stay within a conservative band around recent observations.
    # This prevents extreme values and ensures stability, crucial for random-walk data.
    clip_context = context[-13:] if n >= 13 else context
    
    # Handle cases where clip_context might be empty or all NaNs, to get valid min/max.
    lo = float(np.min(clip_context)) if len(clip_context) > 0 and not np.all(np.isnan(clip_context)) else last
    hi = float(np.max(clip_context)) if len(clip_context) > 0 and not np.all(np.isnan(clip_context)) else last

    # Ensure lo and hi are valid numbers, if they became NaN from an all-NaN clip_context.
    if np.isnan(lo): lo = last
    if np.isnan(hi): hi = last
    if np.isinf(lo): lo = last
    if np.isinf(hi): hi = last

    # Ensure hi is not less than lo.
    if hi < lo: hi = lo

    span = hi - lo

    final_forecast = np.copy(blended_forecast) # Work on a copy

    # If recent data is effectively constant (span is zero or very small), clip to that constant value.
    if span <= 1e-6:
        final_forecast = np.clip(final_forecast, lo, hi)
    else:
        # Otherwise, clip within historical min/max +/- a tighter margin (0.15*span vs 0.25*span in parent).
        # This keeps forecasts within sensible data-driven boundaries for random-walk like data.
        final_forecast = np.clip(final_forecast, lo - 0.15 * span, hi + 0.15 * span)
    
    # Final robust handling of NaN/Inf values:
    # Ensure no NaN or Inf values are returned. NaNs are replaced by the last valid value.
    # Infs are clipped to the recent min/max (lo/hi).
    nan_replacement_value = last
    if np.isnan(last):
        valid_context_values = context[~np.isnan(context)]
        if len(valid_context_values) > 0:
            nan_replacement_value = float(valid_context_values[-1])
        else:
            nan_replacement_value = 0.0 # Fallback if all context values are NaN

    return np.nan_to_num(final_forecast, nan=nan_replacement_value, posinf=hi, neginf=lo).astype(float)