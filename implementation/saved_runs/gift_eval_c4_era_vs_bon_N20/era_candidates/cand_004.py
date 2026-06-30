import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Handle very short contexts by returning naive last-value forecast
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a short-term, recent slope
    # Use the last 4 points for slope calculation
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Calculate scale (standard deviation) for clipping
    # Use the last 13 points if available, otherwise the entire context
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Handle cases where scale might be zero (e.g., all recent values are identical)
    # In such cases, the slope should be zero as there's no variation
    if scale == 0:
        clipped_slope = 0.0
    else:
        # Clip the slope to a small fraction of the recent scale
        # This makes the trend correction very conservative
        clipped_slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Damping factor for the trend
    # A value of 0.6 means the trend effect diminishes quickly over the horizon
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)

    # Generate forecasts starting from the last observed value, adding damped trend
    out = last + clipped_slope * np.cumsum(phi ** steps)

    # Calculate min/max of recent observations for clipping the forecast output
    # Use the last 13 points if available, otherwise the entire context
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Calculate the span (range) of recent values
    span = hi - lo
    
    # If span is zero (e.g., all recent values are identical), define a minimal span
    # to avoid division by zero or overly tight clipping bounds if we added a margin later.
    # However, for the current clipping method (lo - 0.25*span), if span is 0,
    # the bounds become lo and hi, which is correct. So, no special handling needed for span==0.
    
    # Clip the forecasts to stay within a band around recent observations
    # This prevents forecasts from running away due to small trend extrapolations
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Robustly handle any remaining NaN/Inf values, replacing them with sane fallbacks
    # NaN replaced by last value, posinf by hi, neginf by lo
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)