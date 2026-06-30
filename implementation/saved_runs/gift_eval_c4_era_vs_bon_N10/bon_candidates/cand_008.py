import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback for very short series (less than 4 points for trend calculation)
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate recent trend (slope) from the last 4 observations
    # This ensures `recent` always has at least 2 points for `np.diff`
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Calculate a scale for robust clipping of the slope
    # Use last 13 observations if available, otherwise use the full context
    # Hard requirement 9: Indexing safety
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))

    # Guard against zero scale (e.g., if all recent values are identical)
    # If scale is 0, slope will be clipped to 0, making the forecast flat (naive).
    if scale == 0:
        slope = 0.0
    else:
        # Clip the slope to a small fraction of the recent scale
        # This prevents aggressive trend extrapolation, as per recommendations.
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damped trend (Exponentially Damped Trend - EDT)
    # phi = 0.6 is a conservative damping factor
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    # The cumulative sum `cumsum(phi ** steps)` correctly implements the accumulated damped trend
    # for each forecast step: h_1 = phi^1, h_2 = phi^1+phi^2, etc.
    forecasts = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations
    # Use last 13 observations if available, otherwise use the full context
    # Hard requirement 9: Indexing safety
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    span = hi - lo

    # If span is zero (e.g., all historical values are identical),
    # clipping range becomes [lo, hi], which means [value, value].
    # In this case, slope would also be 0, so forecasts would be `last`,
    # which would be correctly clipped to `[last, last]`.
    forecasts = np.clip(forecasts, lo - 0.25 * span, hi + 0.25 * span)

    # Hard requirement 4: Never output NaN or inf.
    # Use nan_to_num to convert any potential NaN, inf, or -inf to safe values.
    # NaN values will be replaced by the last observed value.
    # Pos/Neg infinity values will be clipped to the observed hi/lo range.
    final_output = np.nan_to_num(forecasts, nan=last, posinf=hi, neginf=lo).astype(float)

    # Hard requirement 3: Output length must be EXACTLY prediction_length
    return final_output
