import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # Get the last observed value, which serves as the backbone for the forecast.
    last = float(context[-1])

    # For very short series (less than 4 points), calculating a meaningful trend
    # can be unstable or misleading. In these cases, revert to a simple Naive forecast
    # (repeat the last observed value).
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a short-term trend (slope) from the most recent observations.
    # Using the last 4 data points for slope calculation, as suggested for conservative trends.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale (standard deviation) from recent data to bound the trend correction.
    # Using the last 13 points if available, otherwise the entire context. This window helps
    # capture recent variability without relying on potentially outdated patterns.
    scale_window = context[-13:] if n >= 13 else context
    scale = float(np.std(scale_window))
    
    # If the scale is zero (e.g., the series is completely flat), the slope must also be zero
    # to prevent artificial trend application. Otherwise, clip the slope to a small fraction
    # of the recent standard deviation (10%). This is a strong constraint to prevent
    # aggressive and potentially inaccurate trend extrapolation.
    if scale == 0:
        slope = 0.0
    else:
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Define a damping factor (phi) for the trend. A value of 0.6 ensures the trend effect
    # diminishes quickly over the forecast horizon, making forecasts revert towards 'last'.
    phi = 0.6
    
    # Calculate the cumulative damped trend effect for each step in the prediction horizon.
    # The trend contribution for step `h` is `slope * phi^h`. `cumsum` applies this incrementally.
    steps = np.arange(1, prediction_length + 1)
    damped_trend_effect = slope * np.cumsum(phi ** steps)
    
    # Initial forecast: last observed value plus the cumulative damped trend.
    out = last + damped_trend_effect

    # Apply robust clipping to ensure forecasts remain within a sensible range.
    # The clipping bounds are derived from the min/max of recent observations (last 13 points
    # or entire context), extended by a small margin (25% of the recent data span).
    bounds_window = context[-13:] if n >= 13 else context
    lo = float(np.min(bounds_window))
    hi = float(np.max(bounds_window))
    
    # Calculate the span of recent data. Handle cases where lo equals hi (flat series) to avoid
    # division by zero or incorrect span calculations if it were used in a ratio.
    span = hi - lo if lo != hi else 0.0
    
    # Clip the forecasts to the defined bounds. This prevents forecasts from diverging too far
    # from the observed historical range, adding another layer of conservatism.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Ensure the final output contains no NaN or infinite values.
    # NaN values are replaced with 'last', positive infinity with 'hi', and negative infinity with 'lo'.
    # This guarantees a valid numerical output. Finally, ensure the output type is float.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)