import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    if n == 0:
        # If no context, return zeros (or last value, but zeros is safe)
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    if n < 4:
        # For very short series, fall back to simple naive forecast
        return np.full(prediction_length, last, dtype=float)

    # Calculate a damped trend component
    # Use the last 4 points for slope calculation for recent trend sensitivity
    recent_for_slope = context[-min(n, 4):] # Ensure we don't index beyond n
    if len(recent_for_slope) < 2: # Need at least 2 points to calculate diff
        slope = 0.0
    else:
        slope = float(np.mean(np.diff(recent_for_slope)))

    # Calculate scale (standard deviation) from recent context for robust clipping
    # Use last 13 points or full context if shorter
    scale_period = min(n, 13)
    recent_for_scale = context[-scale_period:]
    scale = float(np.std(recent_for_scale))

    # Guard against zero standard deviation for clipping
    # If std is 0, the series is flat, so trend correction should be 0.
    # The clip below handles this by setting bounds to 0.
    # If scale is 0, the clip will become np.clip(slope, 0, 0), forcing slope to 0.
    
    # Clip the slope to a small fraction of the recent scale to prevent runaway trends
    # This is a key conservative step.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply exponential damping to the trend (Holt's damped trend-like)
    # Phi (damping factor) is set conservatively below 0.7
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    # The trend component accumulates with damping
    forecast_values = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations
    clip_period = min(n, 13)
    recent_for_clip = context[-clip_period:]
    lo = float(np.min(recent_for_clip))
    hi = float(np.max(recent_for_clip))
    span = hi - lo

    # If recent data is completely flat (span is 0), provide a minimal span for clipping
    if span == 0:
        # If all recent values are the same, forecast should ideally stay there.
        # However, for safety and to allow a tiny bit of movement if needed,
        # we can create a very small artificial span.
        # But for this problem, the template's behavior of clipping to exact 'last'
        # if span is 0 is likely the most conservative and best.
        # The default behavior of np.clip(out, lo, hi) if lo==hi will force out to lo.
        # So no specific `span==0` handling is needed, as the template handles it well.
        pass # Sticking to template's implicit handling for span=0

    # Clip the forecast values to be within a sensible range based on recent history
    # Add a small margin (0.25 * span) to allow some variation beyond min/max
    forecast_values = np.clip(forecast_values, lo - 0.25 * span, hi + 0.25 * span)

    # Final safeguard: replace any NaN or Inf values, typically by the last observed value
    # or the bounds themselves for posinf/neginf
    forecast_values = np.nan_to_num(forecast_values, nan=last, posinf=hi, neginf=lo).astype(float)

    return forecast_values