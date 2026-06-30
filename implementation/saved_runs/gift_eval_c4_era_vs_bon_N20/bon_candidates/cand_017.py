import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros as a neutral default
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # For very short contexts (less than 4 points), fall back to pure naive (last value)
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend based on recent observations
    # Use the last 4 points to compute the slope
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope, using up to the last 13 points or all available
    scale = float(np.std(context[-min(n, 13):]))
    # If the series is flat, scale will be 0. Avoid division by zero, but here it's fine for clipping bounds.
    # If scale is 0, slope will be clipped to 0, which is correct for flat series.
    
    # Clip the slope to a small fraction of the recent scale to ensure conservatism
    # This is crucial for preventing forecasts from running away
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damping factor to the trend contribution over the prediction horizon
    # A phi of 0.6 means the trend impact decays quickly
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Generate initial forecasts: last value + damped cumulative trend
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations
    # Use the min/max from the last 13 points (or fewer if n < 13)
    lo = float(np.min(context[-min(n, 13):]))
    hi = float(np.max(context[-min(n, 13):]))
    span = hi - lo
    
    # Define clipping bounds with a small margin (0.25 of the recent span)
    # Ensure forecasts don't go too far beyond recent historical extremes
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Handle potential NaN/Inf values that could arise from extreme numerical operations
    # Replace NaNs with the last observed value, and Infs with recent high/low bounds
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)