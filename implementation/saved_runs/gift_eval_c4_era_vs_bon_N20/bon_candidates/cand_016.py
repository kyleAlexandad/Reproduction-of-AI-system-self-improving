import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context arrays by returning zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # For very short contexts (less than 4 points), fall back to the Naive (last value) forecast.
    # This prevents issues with calculating slope or robust statistics on insufficient data.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # --- Calculate a small, damped trend correction ---

    # Use the last 4 observations to compute a simple average slope.
    # np.diff(context[-4:]) computes differences between adjacent points in the last 4.
    # np.mean averages these differences to get a slope.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale based on recent variability (last 13 points or fewer if not available).
    # This scale is used to conservatively clip the calculated slope.
    scale_window_len = min(n, 13)
    scale = float(np.std(context[-scale_window_len:]))

    # Conservatively clip the slope to a small fraction (10%) of the recent scale.
    # This prevents aggressive extrapolation, aligning with random walk-like behavior.
    slope_limit = 0.1 * scale
    slope = float(np.clip(slope, -slope_limit, slope_limit))

    # Damping factor (phi): The trend effect diminishes quickly over the forecast horizon.
    # A value of 0.6 means the trend contribution to future steps decreases rapidly.
    phi = 0.6

    # Generate the sequence of steps for the prediction horizon.
    steps = np.arange(1, prediction_length + 1)

    # Calculate the forecast: last observed value + damped trend component.
    # np.cumsum(phi ** steps) applies the accumulated damped trend.
    out = last + slope * np.cumsum(phi ** steps)

    # --- Robust clipping to observation range ---

    # Get the min/max values from recent history (last 13 points or fewer if not available).
    range_window_len = min(n, 13)
    lo = float(np.min(context[-range_window_len:]))
    hi = float(np.max(context[-range_window_len:]))
    span = hi - lo # Calculate the range (span) of recent values

    # Clip the forecast values to a slightly extended band around recent observations.
    # This prevents forecasts from deviating too far from the observed range.
    clip_margin = 0.25 * span
    out = np.clip(out, lo - clip_margin, hi + clip_margin)

    # --- Final safety checks ---

    # Ensure no NaN or Inf values are present in the final output.
    # NaN values are replaced with 'last', positive Infs with 'hi', and negative Infs with 'lo'.
    # The result is cast to float to maintain consistency.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)