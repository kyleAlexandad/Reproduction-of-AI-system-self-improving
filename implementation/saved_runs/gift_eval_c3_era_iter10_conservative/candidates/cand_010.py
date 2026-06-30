import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context by returning all zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    last = float(context[-1])

    # Handle very short contexts (less than 4 points) by returning naive (last value) forecast.
    # This also ensures 'recent = context[-4:]' is safe as it's only accessed when n >= 4.
    if n < 4:
        return np.full(prediction_length, last, dtype=float)

    # Calculate raw trend (slope) from the last 4 points.
    # This provides a very short-term view of recent direction.
    recent = context[-4:]
    raw_slope = float(np.mean(np.diff(recent)))

    # Determine a scale for damping and clipping.
    # Use the last 13 points if available, otherwise use all available context.
    scale_context = context[-13:] if n >= 13 else context
    
    # Calculate standard deviation. If it's zero (constant series), no trend should be applied.
    scale = float(np.std(scale_context))
    
    clipped_slope = 0.0
    if scale > 0: # Only apply trend logic if there is variability in the data
        # Keep the clip factor from the previous attempt (0.05). It's already very conservative,
        # restricting the raw slope to a small fraction of the series' recent standard deviation.
        clip_factor = 0.05
        clipped_slope = float(np.clip(raw_slope, -clip_factor * scale, clip_factor * scale))

    # CRITICAL IMPROVEMENT: Drastically reduce the blend weight for the trend component.
    # The parent candidate's MASE was slightly worse than the naive baseline, suggesting
    # that even its small trend contribution (with trend_blend_weight=0.1) was, on average,
    # detrimental for this specific random-walk-like dataset.
    # By reducing this weight significantly (e.g., to 0.005), the model becomes 99.5% naive.
    # This ensures that any deviation from the naive forecast is vanishingly small, thereby
    # bringing the overall MASE extremely close to (or potentially marginally better than)
    # the strong naive baseline, adhering to the "it is better to TIE naive than to diverge
    # far above it" principle.
    trend_blend_weight = 0.005 # Reduced from 0.10 in parent

    effective_slope = clipped_slope * trend_blend_weight

    # Keep the damping factor from the previous attempt (0.3). This already ensures
    # the trend effect decays very rapidly over the forecast horizon, aligning with
    # the random walk behavior.
    phi = 0.3

    # Steps for the forecast horizon (1 to prediction_length)
    steps = np.arange(1, prediction_length + 1)

    # Calculate forecasts. The cumulative sum of phi**steps ensures the trend effect
    # diminishes significantly over the forecast horizon.
    out = last + effective_slope * np.cumsum(phi ** steps)

    # Robust clipping of forecasts based on recent historical min/max.
    # Keep the tighter clipping margin from the previous attempt (0.1).
    # This helps keep forecasts within a plausible range derived from recent observations.
    clip_context = context[-13:] if n >= 13 else context
    lo = float(np.min(clip_context))
    hi = float(np.max(clip_context))
    span = hi - lo

    # The clipping logic correctly handles cases where span is zero (constant series)
    # as lo - 0.1 * 0 = lo and hi + 0.1 * 0 = hi, effectively clipping to [last, last].
    out = np.clip(out, lo - 0.1 * span, hi + 0.1 * span)

    # Ensure no NaN or inf values are returned.
    # Fallback to 'last' for NaN, and historical 'hi'/'lo' for inf, providing robustness.
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)