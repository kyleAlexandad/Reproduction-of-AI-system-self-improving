import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Naive forecast (simple last-value repetition)
    # This is the backbone of the forecast, as pure naive is a very strong baseline.
    naive_forecast = np.full(prediction_length, last, dtype=float)

    # If context is very short (less than 4 points), we rely purely on the naive forecast.
    # The trend calculation requires at least 4 points for meaningful recent differences.
    if n < 4:
        # Robust handling of NaN/Inf for the short context case.
        return np.nan_to_num(naive_forecast, nan=last, posinf=last, neginf=last).astype(float)

    # --- Damped Trend Component Calculation ---
    # Calculate a small, damped trend based on recent observations to provide a slight adjustment.
    # Use the last 4 points for slope calculation to capture very recent movement.
    recent = context[-4:]
    slope_raw = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the raw slope. This prevents extreme trend values.
    # Uses last 13 points if available, otherwise the entire context.
    scale_context = context[-13:] if n >= 13 else context
    scale = float(np.std(scale_context))

    slope = 0.0 # Default to no slope if scale is zero or very small
    if scale > 1e-6: # Only allow a trend if there's some variance in the recent data
        # Clip the raw slope to a very small fraction of the recent scale.
        # This makes the trend component extremely conservative, limiting its impact.
        slope = float(np.clip(slope_raw, -0.075 * scale, 0.075 * scale))
    # If scale is 0 or very small, slope remains 0, meaning no trend is applied.

    # Apply a damping factor to the trend. Its influence diminishes over the forecast horizon.
    # A phi of 0.5 ensures rapid damping, making the trend effect short-lived.
    phi = 0.5 
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the incremental damped trend component for each step.
    # This component will be added to the last observed value.
    damped_trend_component = slope * np.cumsum(phi ** steps)

    # --- Ensemble Blending ---
    # Blend the naive forecast with the damped trend component.
    # Given that pure naive is extremely strong for this task (random walk behavior),
    # and the parent candidate's MASE was slightly worse than naive, it implies
    # that even a small trend component can be detrimental.
    # To improve, we make the model even more conservative by increasing the weight
    # of the naive forecast further (from 0.95 to 0.99), thus reducing the trend's influence
    # to an almost negligible level (from 0.05 to 0.01).
    # This pushes the model closer to pure naive, aiming to match or slightly outperform it.
    weight_naive = 0.99 
    weight_damped_trend_influence = 1.0 - weight_naive # Now 0.01

    # The blended forecast is essentially the last value plus a very small, damped trend correction.
    blended_forecast = last + weight_damped_trend_influence * damped_trend_component

    # --- Robust Clipping ---
    # Clip forecasts to stay within a conservative band around recent observations.
    # This prevents extreme values and ensures stability, crucial for random-walk data.
    clip_context = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_context))
    hi = float(np.max(clip_context))
    span = hi - lo

    final_forecast = np.copy(blended_forecast) # Work on a copy

    # If recent data is constant (span is zero or very small), clip to that constant value.
    if span <= 1e-6:
        final_forecast = np.clip(final_forecast, lo, hi)
    else:
        # Otherwise, clip within historical min/max +/- 25% of the range.
        # This keeps forecasts within a sensible data-driven boundary.
        final_forecast = np.clip(final_forecast, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final robust handling of NaN/Inf values:
    # Ensure no NaN or Inf values are returned. NaNs are replaced by the last valid value,
    # or 0.0 if no valid past values exist. Infs are clipped to recent min/max.
    nan_replacement_value = last
    if np.isnan(last):
        valid_context_values = context[~np.isnan(context)]
        if len(valid_context_values) > 0:
            nan_replacement_value = float(valid_context_values[-1])
        else:
            nan_replacement_value = 0.0

    return np.nan_to_num(final_forecast, nan=nan_replacement_value, posinf=hi, neginf=lo).astype(float)