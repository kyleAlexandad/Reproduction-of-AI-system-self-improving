import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback to pure naive (last value) forecast for very short series
    # (e.g., less than 4 points for reliable slope estimation).
    # This is consistent with the provided conservative template.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # --- Damped Trend Calculation ---
    # Calculate slope from the most recent 4 data points.
    # This window is chosen for quick reactivity to recent changes.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for the data (standard deviation of recent values).
    # This helps in robustly clipping the slope and defining forecast bounds.
    scale_context = context[-13:] if n >= 13 else context
    scale = float(np.std(scale_context))

    # Robustly clip the slope: it should only be a tiny fraction of the data's scale.
    # This prevents aggressive trend extrapolation, aligning with "random walk" behavior.
    if scale == 0:
        # If the recent context values are all identical, the scale is 0, so slope must be 0.
        slope = 0.0
    else:
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Damping factor: `phi` ensures the trend's influence diminishes quickly over the horizon.
    # A value of 0.6 is quite aggressive, making the forecast converge quickly to the baseline.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the cumulative damped trend effect.
    # This is the *correction* term that would be added to the last value.
    damped_trend_correction = slope * np.cumsum(phi ** steps)
    
    # --- Conservative Blending Strategy ---
    # The prompt recommends ensembles/blends where NAIVE keeps a HIGH weight (e.g., 0.75-0.9).
    # We will blend the pure naive forecast (repeating 'last') with the 'last + damped_trend_correction' forecast.
    # A naive_weight of 0.8 means the final forecast is 80% naive and 20% damped trend.
    naive_weight = 0.8 
    
    # The blended forecast is effectively:
    # last + (1 - naive_weight) * damped_trend_correction
    # This reduces the impact of the damped trend significantly.
    blended_forecast = last + (1 - naive_weight) * damped_trend_correction

    # --- Robust Clipping of the Final Forecast ---
    # Define a range based on recent observations to keep forecasts plausible.
    # Use the last 13 points or full context if shorter.
    clip_context = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_context))
    hi = float(np.max(clip_context))
    span = hi - lo

    # Apply clipping to ensure forecasts stay within a realistic band.
    if span == 0:
        # If all recent context values are identical, clip strictly to that value.
        final_out = np.clip(blended_forecast, lo, hi)
    else:
        # Otherwise, allow a small margin (0.25 * span) beyond the observed min/max.
        final_out = np.clip(blended_forecast, lo - 0.25 * span, hi + 0.25 * span)
        
    # Final safety check: replace any NaN/Inf values.
    # NaNs are replaced by the last observed value. Infs are clipped to the observed range.
    return np.nan_to_num(final_out, nan=last, posinf=hi, neginf=lo).astype(float)