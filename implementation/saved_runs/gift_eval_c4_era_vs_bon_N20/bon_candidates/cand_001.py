import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle very short contexts (including empty)
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last_value = float(context[-1])

    # For extremely short contexts, fall back to pure naive (last value)
    # The prompt suggests n < 4 for this fallback.
    if n < 4:
        return np.full(prediction_length, last_value, dtype=float)

    # Calculate a damped trend for longer contexts
    # Use the last 4 points for slope calculation, as suggested for robustness.
    recent_for_slope = context[-4:]
    slope = float(np.mean(np.diff(recent_for_slope)))

    # Determine a scale for clipping the slope and forecast range.
    # Use a longer history (13 points) if available, otherwise the entire context.
    # This helps in handling cases where the context is shorter than 13.
    scale_data = context[-13:] if n >= 13 else context[:]
    scale = float(np.std(scale_data))

    # If the series is perfectly flat, std will be 0. Avoid division by zero issues
    # and ensure slope clipping works correctly by setting scale to a small value if 0.
    if scale == 0:
        scale = 1.0 # Or epsilon, but 1.0 is safe for multiplicative factors.

    # Clip the slope to a small fraction of the data's standard deviation
    # This prevents aggressive trend extrapolation, keeping it conservative.
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Apply a damping factor to the trend extrapolation (phi <= 0.7 recommended)
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    # Calculate forecasts as last_value + damped cumulative trend
    forecasts = last_value + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts within a band around recent observations.
    # Use the last 13 points for min/max if available, otherwise the entire context.
    clip_data = context[-13:] if n >= 13 else context[:]
    lo = float(np.min(clip_data))
    hi = float(np.max(clip_data))
    span = hi - lo

    # If the series is perfectly flat (span is 0), then lo == hi == last_value.
    # Clipping to [lo, hi] would constrain forecasts to last_value.
    # If there's a span, allow a small margin outside the historical range.
    if span == 0:
        # If all historical values are the same, forecasts should stay at that value.
        forecasts = np.clip(forecasts, lo, hi)
    else:
        forecasts = np.clip(forecasts, lo - 0.25 * span, hi + 0.25 * span)

    # Ensure no NaN or Inf values are returned.
    # If any NaNs somehow sneak through (highly unlikely after clipping), replace with last_value.
    # Replace posinf with hi, neginf with lo, though clipping should prevent these too.
    return np.nan_to_num(forecasts, nan=last_value, posinf=hi, neginf=lo).astype(float)