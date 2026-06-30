import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle extremely short contexts (length 0 or 1-3)
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    if n < 4:
        # For very short series, fall back to simple naive (last value)
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend component
    # Use recent 4 points for slope estimation
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope
    # Use 13 recent points if available, otherwise use all available context
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    # Handle case where scale might be 0 (e.g., all recent values are the same)
    if scale < 1e-6: # Use a small epsilon to avoid division by zero or near-zero scale issues
        scale = 1.0 # Default to 1 if there's no variance, effectively removing slope clipping based on scale

    # Clip the slope to be a small fraction of the recent scale
    # This prevents the forecast from running away too quickly
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply damping to the trend over the prediction horizon
    phi = 0.6  # Damping factor
    steps = np.arange(1, prediction_length + 1)
    # The forecast is the last value plus the cumulative damped trend
    out = last + slope * np.cumsum(phi ** steps)

    # Apply robust clipping to keep forecasts within a reasonable band
    # Use 13 recent points for min/max if available, otherwise use all available context
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    span = hi - lo

    # If span is zero (e.g., all context values are identical), adjust clipping range
    if span < 1e-6:
        out = np.clip(out, last - 0.25 * abs(last) * 0.1, last + 0.25 * abs(last) * 0.1) # Small relative band
    else:
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Ensure no NaN or Inf values are returned
    # Replace NaN with 'last', posinf with 'hi', neginf with 'lo'
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)
