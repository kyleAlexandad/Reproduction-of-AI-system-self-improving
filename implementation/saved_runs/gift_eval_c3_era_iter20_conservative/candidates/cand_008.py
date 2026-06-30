import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Naive forecast (simple last-value repetition)
    naive_forecast = np.full(prediction_length, last, dtype=float)

    # If context is very short (less than 4 points), we rely purely on the naive forecast.
    # The trend calculation requires at least 4 points (for context[-4:] and np.diff).
    if n < 4:
        # nan_to_num is applied at the end for all cases for robust output.
        # Here, `naive_forecast` should not have NaNs if `last` is valid.
        return np.nan_to_num(naive_forecast, nan=last, posinf=last, neginf=last).astype(float)

    # --- Damped Trend Component ---
    # Calculate a small, damped trend based on recent observations.
    # Use the last 4 points for slope calculation to capture very recent movement.
    recent = context[-4:]
    slope_raw = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope.
    # Use last 13 points if available, otherwise use the entire context.
    # This prevents extreme slope values from unduly influencing the forecast.
    scale_context = context[-13:] if n >= 13 else context
    scale = float(np.std(scale_context))

    slope = 0.0 # Default to no slope if scale is zero or very small
    if scale > 1e-6: # Only allow a trend if there's some variance in the recent data
        # Clip the raw slope to a very small fraction of the recent scale.
        # This makes the trend component extremely conservative.
        # Kept the parent's value of 0.075*scale which is tighter than the template's 0.1*scale.
        slope = float(np.clip(slope_raw, -0.075 * scale, 0.075 * scale))
    # If scale is 0 or very small, slope remains 0, meaning no trend is applied.

    # Apply a damping factor to the trend, so its influence diminishes over the horizon.
    # Reduced phi from 0.6 (parent) to 0.5 for faster damping, making the trend less persistent.
    phi = 0.5 
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the damped trend forecast component
    damped_trend_forecast = last + slope * np.cumsum(phi ** steps)

    # --- Ensemble Blending ---
    # Blend the naive forecast with the damped trend forecast.
    # Increased weight for naive forecast from 0.9 (parent) to 0.95.
    # This makes the overall forecast even more conservative and closer to pure naive,
    # as the data is stated to behave close to a random walk where naive is very strong.
    weight_naive = 0.95 
    weight_damped_trend = 1.0 - weight_naive # 0.05 weight to the damped trend
    
    blended_forecast = weight_naive * naive_forecast + weight_damped_trend * damped_trend_forecast

    # --- Robust Clipping ---
    # Keep forecasts within a conservative band around recent observations to prevent
    # forecasts from diverging excessively, especially for random-walk like series.
    clip_context = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_context))
    hi = float(np.max(clip_context))
    span = hi - lo

    final_forecast = np.copy(blended_forecast) # Work on a copy

    # If the recent span is zero or very small (i.e., recent values are constant),
    # clip the forecast to that constant value.
    if span <= 1e-6:
        final_forecast = np.clip(final_forecast, lo, hi)
    else:
        # Otherwise, clip the forecast within a band of recent min/max +/- 25% of the span.
        final_forecast = np.clip(final_forecast, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final robust handling of NaN/Inf values:
    # - NaNs are replaced by the last observed value (or 0.0 if last is NaN/invalid).
    # - Positive infinity values are clipped to the recent maximum.
    # - Negative infinity values are clipped to the recent minimum.
    # This ensures the output is always valid floating-point numbers.
    
    # Ensure a valid 'nan' replacement in case 'last' itself is NaN
    nan_replacement_value = last
    if np.isnan(last):
        # Fallback to 0.0 or a valid value from context if last is NaN
        valid_context_values = context[~np.isnan(context)]
        if len(valid_context_values) > 0:
            nan_replacement_value = float(valid_context_values[-1])
        else:
            nan_replacement_value = 0.0

    return np.nan_to_num(final_forecast, nan=nan_replacement_value, posinf=hi, neginf=lo).astype(float)