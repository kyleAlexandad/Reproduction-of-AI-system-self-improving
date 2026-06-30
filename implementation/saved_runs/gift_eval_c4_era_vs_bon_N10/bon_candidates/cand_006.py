import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short contexts gracefully.
    # If no data, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    last = float(context[-1])
    
    # If context is too short for trend calculation (e.g., less than 4 points for diff),
    # revert to simple naive forecast (last value).
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend.
    # Use the last 4 points to estimate a recent slope.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope and output.
    # Use the standard deviation of the last 13 points if available, otherwise use all points.
    # This helps bound the allowed trend change relative to recent variability.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # Prevent issues if scale is zero (e.g., flat series)
    # If scale is zero, it means the data is perfectly flat. In such a case,
    # the slope should also be zero, and forecasts should be flat.
    if scale < 1e-9: # A tiny epsilon to handle floating point near-zeros
        scale = 1.0 # Use a default scale to prevent division by zero in clipping, though slope will be 0 anyway

    # Clip the slope to a small fraction of the scale. This heavily dampens the trend.
    # This prevents aggressive trend extrapolation, keeping forecasts close to naive.
    slope_limit = 0.1 * scale
    slope = float(np.clip(slope, -slope_limit, slope_limit))

    # Apply a heavily damped trend over the forecast horizon.
    # phi = 0.6 ensures the trend effect diminishes quickly.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    # The trend component accumulates, but each step's contribution is damped.
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations.
    # Use the min/max of the last 13 points (or all points if fewer) to define the band.
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    # Calculate the span of recent values. If span is zero, lo and hi are identical.
    span = hi - lo

    # Clip output predictions to be within recent range +/- a small margin.
    # This acts as a safety net against forecasts running wild.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Final robust handling of any potential NaNs, Infs by mapping them to safe values.
    # NaNs default to the last observed value, posinf to the recent max, neginf to the recent min.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)