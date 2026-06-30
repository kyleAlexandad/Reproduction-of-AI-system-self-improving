import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return all zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback for very short series (not enough points for trend estimation).
    # In this case, use the naive (last value) forecast.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a recent, damped trend.
    # Use the last 4 points to estimate the slope, as per recommendations for stability.
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope and later the forecasts.
    # Use the standard deviation of the most recent 13 observations (or all if less than 13).
    # This prevents large, unstable trends from dominating the forecast.
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    # Handle the case where the series is constant (std dev is 0).
    # If scale is 0, slope will be clipped to 0, which is correct for a constant series.
    # No explicit `if scale == 0: scale = 1.0` is needed here as `0.1 * scale` becomes 0.

    # Clip the slope to a small fraction of the series' scale.
    # This ensures any trend correction is very conservative.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damped trend extrapolation.
    # A damping factor (phi < 1) ensures the trend effect diminishes over the prediction horizon,
    # making forecasts converge towards the last observed value.
    phi = 0.6  # Conservative damping factor
    steps = np.arange(1, prediction_length + 1)
    # The trend correction is `slope * (phi^1 + phi^2 + ... + phi^k)` for k-th step.
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations.
    # This acts as a safety net against extreme forecasts.
    # Use min/max of the most recent 13 observations (or all if less than 13).
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    span = hi - lo

    # If the series is constant (span is 0), the clipping band effectively becomes [lo, hi],
    # which is `[value, value]`, correctly forcing `out` to `value`.
    # Add a small margin (0.25 * span) to the min/max bounds.
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Final robust handling: replace any NaNs or Infs that might arise in extreme edge cases.
    # NaNs are replaced by the last observed value.
    # Positive infinity values are capped at 'hi', negative infinity values at 'lo'.
    # Ensure the final output array is of type float.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)