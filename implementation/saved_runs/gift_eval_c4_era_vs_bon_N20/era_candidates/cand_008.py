import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # Get the last observed value, which serves as the backbone for naive forecast
    last = float(context[-1])
    
    # Handle very short contexts (less than 4 points):
    # Fallback to pure naive (last-value) forecast, as trend calculation might be unstable
    if n < 4:
        return np.full(prediction_length, last, dtype=float)
    
    # Calculate a short-term trend (slope)
    # Use the last 4 points for slope calculation to capture very recent changes
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent))) # Mean difference of last 4 points

    # Calculate a scale to normalize and clip the slope
    # Use the standard deviation of the last 13 points (or full context if shorter)
    # This helps in preventing the slope from being too aggressive relative to recent variability
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Clip the slope to a very small fraction (5%) of the recent scale, as in the parent.
    # Even this small trend might have been detrimental for the parent, but we will
    # further reduce its impact via blending.
    slope = float(np.clip(slope, -0.05 * scale, 0.05 * scale))
    
    # Damping factor for the trend
    phi = 0.6
    
    # Calculate steps for the cumulative sum of the damped trend
    steps = np.arange(1, prediction_length + 1)
    
    # 1. Generate a pure naive forecast (last value repeated)
    naive_forecast = np.full(prediction_length, last, dtype=float)
    
    # 2. Generate a damped trend forecast (before final clipping)
    damped_trend_forecast = last + slope * np.cumsum(phi ** steps)

    # Blend the naive forecast and the damped trend forecast.
    # Given that the parent's MASE was worse than naive, we need to be even more
    # conservative. A very high weight for the naive forecast directly addresses
    # the recommendation to keep naive as the anchor with high weight.
    # Let's use 95% weight for naive and 5% for the damped trend.
    W_NAIVE = 0.95
    W_TREND = 1.0 - W_NAIVE # Should be 0.05

    # Combine the forecasts
    out = W_NAIVE * naive_forecast + W_TREND * damped_trend_forecast
    
    # Apply robust clipping to the blended forecasts
    # Determine the minimum and maximum of recent context (last 13 points or full context)
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Calculate the span (range) of recent values
    span = hi - lo

    # Apply robust clipping: keep forecasts within recent min/max +/- a small margin (0.25 * span)
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final safety check: replace any NaN or Inf with sane values
    # NaN values are replaced with 'last' (a robust default)
    # Positive Infs are replaced with 'hi' (the upper bound of recent data)
    # Negative Infs are replaced with 'lo' (the lower bound of recent data)
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)