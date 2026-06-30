import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros for all predictions
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # For very short series (less than 4 points), fallback to a simple naive forecast
    # (repeating the last observed value). This prevents issues with
    # calculating trend or scale on insufficient data.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate a conservative, damped trend component.
    # We use the last 4 points to estimate the recent slope.
    # np.diff on 4 points results in 3 differences, whose mean gives the average change.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for robustly clipping the slope.
    # Use the last 13 points if available, otherwise use the entire context.
    # This helps ensure the scale is relevant to recent history but doesn't overfit to very short series.
    scale_window = context[-13:] if n >= 13 else context
    scale = float(np.std(scale_window))

    # Apply strong clipping to the calculated slope.
    # This is a critical step to prevent aggressive trend extrapolation,
    # as weekly series often behave like a random walk.
    # The trend correction is limited to 10% of the recent standard deviation per step.
    # If scale is 0 (e.g., all recent values are identical), slope will be clipped to 0.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
    
    # Damping factor (phi): The influence of the trend diminishes over time.
    # A value of 0.6 means the trend effect reduces quickly, making forecasts
    # revert towards the last observed value.
    phi = 0.6
    
    # Calculate the cumulative sum of damped trend contributions for each forecast step.
    # (phi ** steps) creates a series like [phi^1, phi^2, ..., phi^prediction_length]
    # np.cumsum accumulates these, so the total trend contribution grows but at a decreasing rate.
    steps = np.arange(1, prediction_length + 1)
    damped_trend_cumulative = np.cumsum(phi ** steps)

    # Generate initial forecasts: Last observed value + damped trend component.
    out = last + slope * damped_trend_cumulative

    # Apply robust clipping to the final forecasts.
    # This keeps forecasts within a sensible band around recent observations,
    # preventing them from straying unrealistically far.
    # Use the last 13 points for min/max range if available, otherwise the entire context.
    recent_for_bounds = context[-13:] if n >= 13 else context
    lo = float(np.min(recent_for_bounds))
    hi = float(np.max(recent_for_bounds))
    span = hi - lo

    # Clip forecasts: values are kept within [recent_min - 0.25*span, recent_max + 0.25*span].
    # If span is 0 (all recent values are identical), this clips `out` to `lo` (which is `hi`).
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Final safeguard: replace any NaN or Inf values that might have occurred (though unlikely
    # with the clipping above) with reasonable fallbacks.
    # NaNs become the `last` observed value. Infs are clipped to `hi` or `lo`.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)