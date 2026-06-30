import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last_value = float(context[-1])

    # Fallback for very short context arrays, where slope calculation might be unstable
    if n < 4:
        return np.full(prediction_length, last_value, dtype=float)

    # Calculate a simple, recent slope
    # Use the last 4 points for slope calculation, if available, for robustness.
    recent_context_for_slope = context[-4:]
    # np.diff needs at least 2 points. With 4 points, diff has 3, and mean is fine.
    slope = float(np.mean(np.diff(recent_context_for_slope)))

    # Determine a scale for clipping the slope and forecasts
    # Use the last 13 points if available, otherwise the full context.
    # This helps in preventing extreme predictions based on overall history if it's very long.
    scale_data = context[-13:] if n >= 13 else context
    scale = float(np.std(scale_data))

    # Guard against zero standard deviation (e.g., if context is flat)
    if scale == 0:
        # If the series is flat, the slope should be zero.
        # The np.clip below will correctly set slope to 0 if scale is 0.
        # But if the series is flat, predictions should just be the last value.
        return np.full(prediction_length, last_value, dtype=float)

    # Clip the slope to a small fraction of the recent standard deviation
    # This ensures any trend correction is very small and damped.
    max_slope_change = 0.1 * scale
    slope = float(np.clip(slope, -max_slope_change, max_slope_change))

    # Apply a heavily damped trend
    # A damping factor (phi) < 0.7 is recommended for conservatism.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    # The cumulative sum ensures the *total* effect of the slope diminishes over time,
    # and the individual step increments also diminish.
    forecast_values = last_value + slope * np.cumsum(phi ** steps)

    # Robust clipping of forecasts within a band around recent observations
    # Get min/max from the same recent window used for scale.
    min_recent = float(np.min(scale_data))
    max_recent = float(np.max(scale_data))
    span = max_recent - min_recent

    # Add a small margin to the min/max range
    margin = 0.25 * span
    lower_bound = min_recent - margin
    upper_bound = max_recent + margin

    # If the span is zero (e.g., flat series), the bounds will be min_recent.
    # Clip forecasts to prevent them from straying too far.
    forecast_values = np.clip(forecast_values, lower_bound, upper_bound)

    # Ensure no NaN or Inf values are returned
    # Fallback for NaN: last_value, posinf: max_recent, neginf: min_recent
    forecast_values = np.nan_to_num(forecast_values, nan=last_value, posinf=max_recent, neginf=min_recent)

    return forecast_values.astype(float)