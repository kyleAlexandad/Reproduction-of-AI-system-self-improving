import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback to naive (last value) for very short contexts (less than 4 points)
    # This ensures that calculations requiring a history of 4 points or more are safe.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # --- Trend Component Calculation ---
    # Use the last 4 points to estimate the recent slope.
    # np.diff computes differences between consecutive elements.
    # np.mean of these differences gives an average slope over the recent window.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for normalizing and clipping the slope.
    # Use the standard deviation of up to the last 13 points (or all available context if n < 13).
    # This helps to make the slope correction relative to the recent variability.
    scale_period = 13
    if n >= scale_period:
        scale = float(np.std(context[-scale_period:]))
    else:
        scale = float(np.std(context))
    
    # Handle the case where scale is zero (e.g., if context is flat)
    # In this scenario, the slope should be zero, effectively reverting to naive.
    # The clip ensures the slope is a 'tiny correction' as recommended.
    max_slope_factor = 0.1 # Cap slope to 10% of recent standard deviation
    slope = float(np.clip(slope, -max_slope_factor * scale, max_slope_factor * scale))

    # --- Damped Trend Forecast ---
    # `phi` is the damping factor (0 < phi <= 0.7). A lower phi means quicker damping.
    # 0.6 is used here for a conservative approach.
    phi = 0.6 
    
    # `steps` are the forecast steps (1, 2, ..., prediction_length)
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate the forecast: last value + cumulative sum of damped slope contributions.
    # The term `phi ** steps` ensures that the trend contribution diminishes over time.
    out = last + slope * np.cumsum(phi ** steps)

    # --- Robust Clipping ---
    # Keep forecasts within a safe band defined by recent observations.
    # Determine the recent minimum and maximum values using up to the last 13 points.
    clip_period = 13
    if n >= clip_period:
        lo = float(np.min(context[-clip_period:]))
        hi = float(np.max(context[-clip_period:]))
    else:
        lo = float(np.min(context))
        hi = float(np.max(context))
    
    span = hi - lo # Range of recent values

    # Clip the forecast to be within a band around the recent min/max.
    # Adding a small margin (0.25 * span) allows for slight extrapolation but prevents wild values.
    # If span is 0 (flat history), the clipping range becomes [lo, hi], i.e., [value, value].
    clip_margin_factor = 0.25 
    out = np.clip(out, lo - clip_margin_factor * span, hi + clip_margin_factor * span)

    # --- Final Safeguard against NaN/Inf ---
    # Replace any NaN values in the output with the last observed value.
    # Replace positive infinity with the recent maximum (hi) and negative infinity with recent minimum (lo).
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)