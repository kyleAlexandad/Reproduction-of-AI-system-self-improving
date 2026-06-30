import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # HARD REQUIREMENT 8: Handle SHORT context arrays robustly
    # If the context is empty, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # For very short contexts (less than 4 points), fall back to naive (last value repeated).
    # This prevents unstable trend calculations and ensures safety for small 'n'.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a simple trend (slope) from the most recent 4 observations.
    # This is only executed if n >= 4, making context[-4:] safe.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine the window for calculating statistics (standard deviation, min, max).
    # Use the last 13 observations if available, otherwise use all available context.
    # This ensures a reasonable window size for robust statistics while being safe for n < 13.
    stats_window = context[-13:] if n >= 13 else context

    # Calculate the scale (standard deviation) of the stats_window.
    # This scale is used to bound the slope, preventing aggressive extrapolation.
    scale = float(np.std(stats_window))

    # Clip the calculated slope to a small fraction of the scale.
    # This is a critical step for conservatism, preventing forecasts from running away.
    # If 'scale' is 0 (e.g., all values in stats_window are identical), 'slope' will be clipped to 0.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damped trend. 'phi' is the damping factor, chosen conservatively (0.6 <= 0.7).
    # The cumulative sum of phi**steps applies decreasing influence of the trend over time.
    phi = 0.6
    steps = np.arange(1, prediction_length + 1) # Steps from 1 to prediction_length
    
    # Calculate the forecast points based on the last observed value plus the damped trend.
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping of forecasts within a band around recent observations.
    # This prevents forecasts from becoming unrealistic or diverging excessively.
    lo = float(np.min(stats_window))
    hi = float(np.max(stats_window))
    span = hi - lo

    # The clipping range extends slightly beyond the observed min/max values.
    # If 'span' is 0 (all values in stats_window are identical), the range becomes [lo, hi],
    # effectively clamping 'out' to 'lo' (which is 'last').
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # HARD REQUIREMENT 4: Never output NaN or inf.
    # Final safeguard: replace any potential NaN or Inf values with sensible defaults.
    # NaN values are replaced by 'last'. Positive Inf by 'hi'. Negative Inf by 'lo'.
    # .astype(float) ensures the final output array elements are of float type.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)
