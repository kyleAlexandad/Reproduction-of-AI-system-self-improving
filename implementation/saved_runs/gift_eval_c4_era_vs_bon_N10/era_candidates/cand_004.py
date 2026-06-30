import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # 8. Handle SHORT context arrays robustly
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    last = float(context[-1])

    # 8. Handle SHORT context arrays robustly (less than 4 points, fall back to naive)
    if n < 4:
        return np.full(prediction_length, last, dtype=float)
    
    # Calculate slope from the most recent 4 points
    # 9. INDEXING SAFETY: context[-4:] is safe because n >= 4 here
    recent = context[-4:]
    slope = float(np.mean(np.diff(recent)))

    # Calculate scale (standard deviation) from recent observations for clipping the slope
    # 9. INDEXING SAFETY: context[-13:] is safe due to n >= 13 check
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    
    # If scale is zero (e.g., all recent values are identical), use a small positive number
    # to avoid division by zero or zero-range clipping if it were used as a divisor.
    # For `0.1 * scale`, if scale is 0, the clipping range is [0, 0], which means slope becomes 0,
    # which is the desired behavior for a flat series. So, no special handling for scale=0 needed here.
    
    # Clip the slope to a small fraction of the recent scale to ensure conservatism
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))

    # Define the damping factor for the trend
    phi = 0.6
    
    # Generate steps for the forecast horizon
    steps = np.arange(1, prediction_length + 1)
    
    # Calculate damped trend extrapolation
    # The term `np.cumsum(phi ** steps)` ensures the trend contribution diminishes over time
    out = last + slope * np.cumsum(phi ** steps)

    # Apply robust clipping based on the min/max of recent observations
    # 9. INDEXING SAFETY: context[-13:] is safe due to n >= 13 check
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    
    span = hi - lo
    # If span is 0 (all context values are the same), then `lo - 0.25 * span` simplifies to `lo`,
    # and `hi + 0.25 * span` simplifies to `hi`. The forecast will be clipped to `last`, which is correct.
    
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)

    # 4. Never output NaN or inf. Replace with sensible fallbacks (last observed value, or bounds)
    out = np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)
    
    # 3. Output length must be EXACTLY prediction_length (already ensured by np.arange and subsequent operations)
    return out