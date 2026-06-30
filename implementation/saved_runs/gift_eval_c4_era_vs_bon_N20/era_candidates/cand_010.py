import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros as a safe fallback
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # Get the last observed value, which serves as the backbone for the forecast.
    last = float(context[-1])
    
    # For very short series (less than 4 points), fall back to a pure naive forecast.
    # This prevents issues with calculating trend from insufficient data.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)
    
    # Calculate a small, damped trend if sufficient data is available (n >= 4).
    # This approach is based on the "GOOD CONSERVATIVE TEMPLATE" provided in the prompt,
    # which recommends adding a small, heavily damped trend as a correction to naive.
    
    # 1. Calculate the slope from the most recent 4 points.
    # Using `np.diff` then `np.mean` gives an average per-step change.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))
    
    # 2. Determine a scale for clipping the slope.
    # Use the standard deviation of the last 13 points if available, otherwise the entire context.
    # This makes the slope correction relative to the recent volatility of the series.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Handle cases where scale might be zero (e.g., if all recent values are identical).
    # This prevents division by zero or issues with clipping if scale is degenerate.
    if scale == 0:
        scale = 1.0 # Use a default scale to allow some potential for slope, or keep slope at 0.
                    # Given the clip range is -0.1*scale to 0.1*scale, if scale is 0, slope will be 0.
                    # If it's very small, scale=1.0 is a reasonable default to allow some movement if needed,
                    # though the subsequent clipping with span might catch this.
                    # For `np.clip(slope, -0.1 * scale, 0.1 * scale)`, if scale=0, slope becomes 0.
                    # This is safe and effectively disables the trend if there's no variance.

    # 3. Clip the slope aggressively. This ensures the trend contribution is very small,
    # preventing forecasts from running away, as advised by the prompt.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
    
    # 4. Apply heavy damping to the trend over the prediction horizon.
    # phi = 0.6 is a value suggested in the template for aggressive damping.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1) # Steps 1 to prediction_length
    
    # Calculate the forecast: last value + cumulative damped trend.
    out = last + slope * np.cumsum(phi ** steps)
    
    # Apply robust clipping to the forecasts. This keeps predictions within a sensible band
    # around recent observations, preventing extreme values.
    
    # Determine the minimum and maximum of recent context (last 13 points or full context).
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Calculate the span (range) of recent values.
    span = hi - lo
    
    # Apply robust clipping: keep forecasts within recent min/max +/- a small margin (0.25 * span).
    # If span is 0 (e.g., all recent values are identical), this clips to `lo` (which is equal to `hi`).
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final safety check: replace any NaN or Inf values in the output array.
    # NaN values are replaced with 'last' (the most recent robust observation).
    # Positive Infs are replaced with 'hi' (the upper bound of recent data).
    # Negative Infs are replaced with 'lo' (the lower bound of recent data).
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)