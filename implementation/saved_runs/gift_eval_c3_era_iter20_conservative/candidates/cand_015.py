import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short context arrays
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # The last observed value is a strong baseline (Naive forecast).
    last = float(context[-1])

    # For very short series (less than 4 points), fall back to simple last-value naive.
    # This prevents unstable calculations of slope or scale from insufficient data.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # --- Damped Trend Calculation ---
    # Use the last 4 points to calculate a recent slope.
    # This window is short to react to recent changes but avoid over-extrapolation from older data.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Calculate a scale (standard deviation) from recent history.
    # This helps normalize the slope correction relative to the series' typical volatility.
    # Use up to 13 points for the scale, or fewer if the context is shorter.
    scale_data_window_length = min(n, 13)
    scale = float(np.std(context[-scale_data_window_length:]))

    # Clip the slope to a small fraction of the scale.
    # This is a crucial step to make the trend correction very conservative and prevent runaway forecasts.
    # If scale is 0 (e.g., context is constant), slope will be clipped to 0, correctly disabling trend.
    slope_clip_factor = 0.1 # Max per-step change is 10% of recent std dev
    slope = float(np.clip(slope, -slope_clip_factor * scale, slope_clip_factor * scale))

    # --- Forecast Generation ---
    # Damping factor (phi). A value of 0.6 means the trend effect diminishes quickly,
    # making later forecasts closer to the naive last value.
    phi = 0.6
    
    # Generate steps for the prediction horizon (1-indexed for convenience in power calculation)
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the cumulative damped trend contribution for each forecast step.
    # The term `phi ** steps` ensures the trend impact decreases over the horizon.
    # `np.cumsum` accumulates this damped trend.
    damped_trend_contribution = slope * np.cumsum(phi ** steps)
    
    # The final forecast is the last observed value plus the damped trend.
    out = last + damped_trend_contribution

    # --- Robust Clipping to Historical Range ---
    # Determine a recent min/max range from historical data to serve as a clipping boundary.
    # This prevents forecasts from going unrealistically high or low.
    clip_data_window_length = min(n, 13)
    lo = float(np.min(context[-clip_data_window_length:]))
    hi = float(np.max(context[-clip_data_window_length:]))
    
    # Calculate the span of recent values.
    span = hi - lo
    
    # Extend the clipping band slightly beyond the historical min/max.
    # If span is 0 (all recent values are the same), the margin will be 0, and forecasts
    # will be clipped to exactly `lo` (which equals `hi`).
    clip_margin_factor = 0.25 # Extend the band by 25% of the span on each side
    
    # Apply the clipping to the generated forecasts.
    out = np.clip(out, lo - clip_margin_factor * span, hi + clip_margin_factor * span)

    # --- Final Robustification ---
    # Replace any potential NaN or inf values, ensuring the output is always finite.
    # NaN values are replaced with the last observed value.
    # Positive/negative infinities are capped at the bounds of the recent history (`hi`/`lo`).
    out = np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)
    
    return out