import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback for very short series, return naive last value
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a recent slope (trend)
    # Using the last 4 points to compute the average difference
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for clipping the slope and forecasting band
    # Use last 13 points if available, otherwise use all context
    if n >= 13:
        scale_context = context[-13:]
    else:
        scale_context = context
    
    # Handle cases where std dev might be zero (e.g., constant series)
    scale = float(np.std(scale_context))
    if scale == 0: # If context is constant, slope must be 0
        slope = 0.0
    else:
        # Clip the slope to a small fraction of the scale to prevent over-extrapolation
        # This is a crucial conservative step
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a heavily damped trend to the forecast
    # phi = 0.6 means the trend contribution quickly fades
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    # The cumulative sum ensures that the trend effect diminishes over time
    out = last + slope * np.cumsum(phi ** steps)

    # Clip forecasts to a reasonable historical band to prevent extreme values
    # Use min/max from last 13 points or all context if shorter
    if n >= 13:
        min_val = float(np.min(context[-13:]))
        max_val = float(np.max(context[-13:]))
    else:
        min_val = float(np.min(context))
        max_val = float(np.max(context))
    
    span = max_val - min_val
    
    # Define a band around the historical min/max values
    # Expand the band slightly (0.25 * span) to allow for some deviation
    lower_bound = min_val - 0.25 * span
    upper_bound = max_val + 0.25 * span
    
    out = np.clip(out, lower_bound, upper_bound)

    # Ensure no NaN or Inf values are returned
    # Replace NaN with 'last', posinf with 'upper_bound', neginf with 'lower_bound'
    return np.nan_to_num(out, nan=last, posinf=upper_bound, neginf=lower_bound).astype(float)