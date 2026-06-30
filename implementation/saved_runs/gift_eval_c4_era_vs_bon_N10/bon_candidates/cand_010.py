import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle SHORT context arrays robustly.
    # If no historical data, return an array of zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Get the last observed value, which is the cornerstone for naive forecasting.
    last = float(context[-1])

    # For very short series (e.g., less than 4 points), fall back to pure naive (last value).
    # This prevents issues with calculating trend from insufficient data.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend for a small correction to the naive forecast.

    # Use the last 4 points to estimate a recent slope.
    # This is safe because n >= 4 at this point.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for the series (standard deviation of recent values)
    # to make the slope clipping robust and relative to the data's variability.
    # Use the last 13 points if available, otherwise use all available context.
    if n >= 13:
        scale_context = context[-13:]
    else:
        scale_context = context
    
    # Calculate the standard deviation. If all values are constant, std will be 0.
    scale = float(np.std(scale_context))

    # Clip the calculated slope to a small fraction of the series' scale.
    # This ensures the trend correction is "tiny" and conservative, preventing forecasts
    # from deviating too much from the last observed value.
    # If scale is 0 (constant series), slope_limit becomes 0, and slope is clipped to 0.
    slope_limit = 0.1 * scale
    slope = float(np.clip(slope, -slope_limit, slope_limit))

    # Apply damping to the trend over the forecast horizon.
    # `phi` is the damping factor (0 < phi <= 0.7 recommended).
    # `np.cumsum(phi ** steps)` ensures the trend contribution diminishes over time.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    
    # Generate initial forecasts: last value plus the damped trend correction.
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: Restrict forecasts within a sensible band around recent observations.
    # This prevents extreme predictions, especially if the trend calculation becomes unstable.
    # Use the min/max of the last 13 points or all context, similar to scale.
    if n >= 13:
        clip_context = context[-13:]
    else:
        clip_context = context

    lo = float(np.min(clip_context))
    hi = float(np.max(clip_context))
    span = hi - lo

    # Define the clipping bounds, allowing forecasts to extend slightly beyond recent min/max.
    # If span is 0 (constant series), clip_lower_bound = lo, clip_upper_bound = hi, effectively pinning forecasts to `last`.
    clip_lower_bound = lo - 0.25 * span
    clip_upper_bound = hi + 0.25 * span
    out = np.clip(out, clip_lower_bound, clip_upper_bound)

    # Hard requirement 4: Never output NaN or inf.
    # Use np.nan_to_num to replace any remaining NaNs or Infs with sensible values.
    # NaNs are replaced by the 'last' observed value.
    # Positive/negative Infs are clipped to the calculated upper/lower bounds.
    return np.nan_to_num(out, nan=last, posinf=clip_upper_bound, neginf=clip_lower_bound).astype(float)