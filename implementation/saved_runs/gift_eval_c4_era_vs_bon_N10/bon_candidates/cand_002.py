import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short or empty contexts
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    last = float(context[-1])
    
    # For contexts with fewer than 4 points, default to naive (last value)
    if n < 4:
        return np.full(prediction_length, last, dtype=float)
    
    # Calculate a damped trend using recent values
    # Use the last 4 points to compute a stable slope
    recent = context[-4:] 
    slope = float(np.mean(np.diff(recent)))

    # Calculate a scale for clipping the slope
    # Use std dev of last 13 points if available, otherwise use all points
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Guard against zero scale to avoid division by zero or overly aggressive clipping
    # If scale is very small, cap slope to a nominal small value or 0
    if scale < 1e-6:
        slope = 0.0
    else:
        # Clip the slope to a small fraction of the recent scale for conservatism
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale)) 

    # Damping factor for the trend
    phi = 0.6 
    
    # Generate steps for cumulative sum of damped trend
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate forecasts: last value + damped cumulative trend
    out = last + slope * np.cumsum(phi ** steps)

    # Apply robust clipping based on historical min/max values
    # Use the last 13 points if available, otherwise use all points for min/max
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Ensure lo <= hi, especially if context is constant or has very few points
    if lo > hi: # Handles cases like context = [5,5,5] or context = [5]
        lo = last - 1e-6 # Set a small range around last
        hi = last + 1e-6
    
    span = hi - lo
    
    # Clip forecasts to a band around historical observations
    # Adding a margin (0.25 * span) allows for slight extrapolation
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    
    # Final check for NaN or inf values, replace with sensible defaults
    # nan: last observed value
    # posinf: historical max (hi)
    # neginf: historical min (lo)
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)