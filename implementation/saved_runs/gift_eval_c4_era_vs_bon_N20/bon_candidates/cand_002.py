import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short series
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # If context is very short, fallback to naive last-value forecast
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend
    # Use the last 4 points to estimate the slope, as per template recommendation
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Calculate scale from recent observations for clipping the slope
    # Use last 13 points if available, otherwise use all context
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Avoid division by zero or NaN issues if scale is zero or NaN
    if np.isnan(scale) or scale == 0:
        # If there's no variance, treat it as a flat series, slope should be zero
        slope = 0.0
    else:
        # Clip the slope to a small fraction of the scale to be conservative
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply damping factor (phi) to the trend
    # A conservative phi (e.g., 0.6) is recommended for damped extrapolation
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Generate initial forecasts with damped trend
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations
    # Use min/max from last 13 points if available, otherwise from all context
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    span = hi - lo
    if span == 0: # Handle cases where all recent values are identical
        # If all recent values are the same, span is 0. Clipping should just enforce that value.
        # Forecasts should ideally already be that value if slope was correctly set to 0.
        out = np.clip(out, lo, hi)
    else:
        # Clip forecasts within a band of 0.25 * span around recent min/max
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Handle potential NaN or Inf values in the output
    # Replace NaNs with 'last', posinf with 'hi', neginf with 'lo' for robustness
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)
