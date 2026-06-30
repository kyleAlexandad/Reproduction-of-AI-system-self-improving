import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = context[-1]

    # Fallback to naive forecast for very short contexts
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate recent slope (trend)
    # Using the last 4 points for slope, as recommended by the template for stability
    recent_for_slope = context[-4:]
    slope = np.mean(np.diff(recent_for_slope))

    # Calculate scale for clipping the slope and forecasts
    # Use last 13 points if available, otherwise use all context
    if n >= 13:
        scale = np.std(context[-13:])
        min_val = np.min(context[-13:])
        max_val = np.max(context[-13:])
    else:
        scale = np.std(context)
        min_val = np.min(context)
        max_val = np.max(context)

    # Handle cases where scale might be zero (e.g., all context values are the same)
    # Default to a small positive value if scale is zero to avoid division by zero or overly aggressive clipping range
    if scale == 0:
        scale = 1.0 # arbitrary small positive value, or could be last * 0.01

    # Clip the slope to prevent aggressive trends, a tiny correction only
    # The factor 0.1 * scale means the per-step change is at most 10% of the recent standard deviation
    slope = np.clip(slope, -0.1 * scale, 0.1 * scale)

    # Apply damped trend forecast
    phi = 0.6  # Damping factor, <= 0.7 as recommended
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate cumulative damped trend for each step
    # out[h] = last + slope * (phi^1 + phi^2 + ... + phi^h)
    damped_trend_component = np.cumsum(phi ** steps)
    out = last + slope * damped_trend_component

    # Robust clipping: keep forecasts within a band around recent observations
    span = max_val - min_val
    
    # Handle span == 0 if all recent values are the same
    if span == 0:
        # If span is zero, all values are the same, so min_val and max_val are equal to last.
        # Clip to `last` itself, effectively making the forecast naive.
        out = np.clip(out, last, last)
    else:
        # Allow a small margin (25% of the span) beyond observed min/max
        out = np.clip(out, min_val - 0.25 * span, max_val + 0.25 * span)

    # Ensure no NaN or Inf values in the output
    # Replace NaN with 'last', positive Inf with 'max_val', negative Inf with 'min_val'
    out = np.nan_to_num(out, nan=last, posinf=max_val, neginf=min_val)

    return out.astype(float)