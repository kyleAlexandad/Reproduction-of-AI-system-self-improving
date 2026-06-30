import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Fallback to naive for very short series (less than 4 points for slope calculation)
    if n < 4:
        # Hard requirement 3: Output length must be EXACTLY prediction_length
        # Hard requirement 4: Never output NaN or inf
        return np.full(prediction_length, last, dtype=float)

    # Calculate a conservative, damped trend
    # Use the last 4 points for the slope, as recommended for robustness against longer trends
    # Hard requirement 9: Indexing safety (context[-4:] is safe)
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Determine a scale for clipping the slope
    # Use standard deviation of recent values (last 13, or all if less than 13)
    # Hard requirement 9: Indexing safety
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))

    # Guard against zero scale (e.g., if all recent values are identical)
    # If scale is 0, slope will be clipped to 0, effectively removing trend.
    if scale == 0:
        slope = 0.0
    else:
        # Clip the slope to a small fraction of the recent scale to ensure tiny corrections only
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Damping factor for the trend, recommended to be <= 0.7.
    # A value of 0.6 makes the trend decay reasonably fast.
    phi = 0.6

    # Generate forecast steps
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate initial forecasts: last value + damped cumulative trend
    out = last + slope * np.cumsum(phi ** steps)

    # Robust clipping: keep forecasts inside a band around recent observations
    # Get min/max of recent values (last 13, or all if less than 13)
    # Hard requirement 9: Indexing safety
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    span = hi - lo

    # Clip output to a band around recent min/max, with a small margin (25% of span)
    # Guard against zero span (e.g., if all recent values are identical)
    if span == 0:
        out = np.clip(out, lo, hi) # If span is zero, clip to that single value
    else:
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # Hard requirement 4: Never output NaN or inf.
    # Replace NaN with the last observed value, and inf with hi/lo bounds.
    # Ensure output is float type.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)