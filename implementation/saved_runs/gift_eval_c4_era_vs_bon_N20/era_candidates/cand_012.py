import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros as a safe fallback.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    last = float(context[-1])
    
    # Handle very short context (e.g., less than 4 points): return a pure naive forecast.
    # This is a conservative strategy to prevent unstable trend calculations from insufficient data.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)
    
    # Calculate a short-term slope from the most recent 4 data points.
    # This localized trend is less likely to extrapolate aggressively.
    # Since we've already checked `n < 4`, `context[-4:]` is guaranteed to have 4 elements.
    slope = float(np.mean(np.diff(context[-4:])))
    
    # Determine a scale for judiciously clipping the slope.
    # Use the standard deviation of the last 13 points if available, otherwise use the entire context.
    # This scale helps bound the magnitude of the trend correction relative to recent variability.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Clip the calculated slope to be a very small fraction (10%) of the recent scale.
    # This is a critical step to ensure the trend correction is "tiny" and "safe",
    # preventing forecasts from diverging too much from the last observed value.
    max_abs_slope = 0.1 * scale
    slope = float(np.clip(slope, -max_abs_slope, max_abs_slope))
    
    # Apply a damped trend to the forecasts.
    # The damping factor `phi=0.6` ensures that the influence of the trend diminishes
    # rapidly as the forecast horizon extends, making the long-term forecast conservative.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1) # Represents forecast steps: 1, 2, ..., prediction_length
    
    # Calculate the cumulative sum of damped factors (phi^1, phi^1+phi^2, etc.)
    # This applies the damped slope incrementally to each forecast step.
    damped_trend_factors = np.cumsum(phi ** steps)
    out = last + slope * damped_trend_factors
    
    # Implement robust clipping of the forecasts.
    # This keeps forecasts within a credible range defined by recent observations,
    # preventing extreme or unrealistic predictions.
    
    # Determine the minimum and maximum values from recent history.
    # Use the last 13 points if available, otherwise use the entire context.
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    span = hi - lo
    
    # Clip the forecasts to be within the recent observed range, plus a small margin.
    # The `0.25 * span` margin allows for minor deviations outside the historical min/max.
    # If `span` is 0 (e.g., constant series), forecasts are effectively clipped to `lo` (which equals `hi`).
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final safety measure: replace any NaN or Inf values that might have occurred
    # (e.g., from division by zero, though guarded against here, or extreme arithmetic)
    # with sensible fallback values (`last`, `hi`, `lo`).
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)
