import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last_value = float(context[-1])

    # Fallback to naive forecast for very short series (less than 4 points)
    # The prompt's template uses 4 as the threshold for trend calculation.
    if n < 4:
        return np.full(prediction_length, last_value, dtype=float)

    # Calculate a conservative, damped trend
    # Use the last 4 points for slope calculation, as suggested by the template
    recent_for_slope = context[-4:]
    
    # Calculate slope. np.diff needs at least 2 points. `recent_for_slope` has 4 if n >= 4.
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for clipping the slope and for output range
    # Use the last 13 points if available, otherwise the entire context
    scale_window_size = 13
    if n >= scale_window_size:
        scale_data = context[-scale_window_size:]
    else:
        scale_data = context

    # Calculate standard deviation. Handle cases where std might be 0 (e.g., constant series).
    scale = float(np.std(scale_data))
    if scale == 0: # If series is constant, slope should be 0.
        slope = 0.0
    else:
        # Clip the slope to a small fraction of the scale (0.1 * std_dev)
        # This prevents aggressive trend extrapolation, keeping it conservative.
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
    
    # Damping factor for the trend. 0.6 is suggested in the template (phi <= 0.7).
    phi = 0.6
    
    # Calculate the steps for prediction horizon
    steps = np.arange(1, prediction_length + 1)
    
    # Apply the damped trend to the last observed value
    forecast_values = last_value + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations
    # Use min/max of the same window as used for scale (last 13 or all context)
    min_val = float(np.min(scale_data))
    max_val = float(np.max(scale_data))
    
    # Handle cases where min_val and max_val might be the same (constant series)
    if min_val == max_val:
        # If the series is constant, the range is 0. Clip to the constant value.
        forecast_values = np.clip(forecast_values, min_val, max_val)
    else:
        span = max_val - min_val
        # Clip within recent min/max +/- 0.25 * span, as suggested by the template
        forecast_values = np.clip(forecast_values, min_val - 0.25 * span, max_val + 0.25 * span)

    # Ensure no NaN or Inf values are returned
    # Default to `last_value` for NaN, `max_val` for posinf, `min_val` for neginf
    # This ensures a clean output even if intermediate calculations resulted in extreme values.
    return np.nan_to_num(forecast_values, nan=last_value, posinf=max_val, neginf=min_val).astype(float)