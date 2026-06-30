import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)

    # 1. Handle very short series
    if len(context) == 0:
        return np.zeros(prediction_length, dtype=float)
    if len(context) == 1:
        # A single point: can't calculate trend or seasonality. Use last value naive.
        return np.repeat(context[-1], prediction_length).astype(float)

    # 2. Parameters
    # Determine seasonal length. Default for 'W' freq is 52 weeks.
    season_length = None
    if metadata is not None:
        season_length = metadata.get('season_length')

    if season_length is None:
        if freq == 'W':
            season_length = 52
        else:
            season_length = 1  # Treat as non-seasonal for other frequencies or if not specified

    # Ensure season_length is at least 1. If 1, it effectively means no seasonality.
    if season_length < 1:
        season_length = 1

    # Smoothing window for the base level calculation.
    # Uses up to the last 5 points for a smoothed representation of the current level.
    smoothing_window = min(len(context), 5)

    # Trend window for calculating the recent trend.
    # Requires at least 2 points to calculate a difference. Uses up to the last 7 points.
    trend_window = max(2, min(len(context), 7))

    # Damping factor for the trend component. Prevents forecasts from escalating/deescalating indefinitely.
    damping_factor = 0.95

    # 3. Calculate Base Level (Smoothed last value)
    # The current level is estimated as the mean of the most recent observations.
    level = np.mean(context[-smoothing_window:])

    # 4. Calculate Trend (Average of recent differences)
    # The trend is estimated as the mean of the differences between consecutive points
    # in the trend window.
    recent_diffs = np.diff(context[-trend_window:])
    trend = np.mean(recent_diffs) if len(recent_diffs) > 0 else 0.0

    forecasts = np.empty(prediction_length, dtype=float)

    # 5. Forecast Generation (Seasonal + Damped Trend or Damped Trend only)
    # A seasonal component can be used only if there's enough historical data.
    # At minimum, we need `season_length` points to look back for a full previous cycle.
    has_sufficient_seasonal_history = len(context) >= season_length

    for h in range(prediction_length):
        # Apply damping to the trend. The trend's influence diminishes over the forecast horizon.
        current_trend_contribution = trend * (h + 1) * (damping_factor ** h)

        if has_sufficient_seasonal_history:
            # Use the value from `season_length` periods ago (Seasonal Naive component)
            # and adjust it with the damped trend.
            # `h % season_length` ensures correct phase if prediction_length > season_length,
            # but for prediction_length=13 and season_length=52, it simplifies to just `h`.
            seasonal_idx_in_context = len(context) - season_length + (h % season_length)
            
            # `seasonal_idx_in_context` is guaranteed to be a valid index if `has_sufficient_seasonal_history` is true
            # and `season_length >= 1`.
            seasonal_base = context[seasonal_idx_in_context]
            forecasts[h] = seasonal_base + current_trend_contribution
        else:
            # If not enough history for seasonality, fall back to a level + damped trend model.
            forecasts[h] = level + current_trend_contribution
            
    # 6. Clipping/Guarding (ensure non-negative forecasts and within a reasonable range)
    # This helps to prevent NaN, inf, or extremely unrealistic values in the output.
    min_val_context = np.min(context)
    max_val_context = np.max(context)

    # If the historical data is all non-negative, ensure forecasts are also non-negative.
    if min_val_context >= 0:
        forecasts = np.maximum(0.0, forecasts)
    
    # Apply dynamic clipping to prevent forecasts from straying too far from the observed range,
    # but allowing for some extrapolation (e.g., 50% beyond the min/max observed).
    if min_val_context != max_val_context: # Only clip if there's variance in the context data
        range_span = max_val_context - min_val_context
        clip_lower = min_val_context - range_span * 0.5
        clip_upper = max_val_context + range_span * 0.5
        
        # If context data is non-negative, ensure the lower clip bound is also non-negative.
        if min_val_context >= 0:
            clip_lower = np.maximum(0.0, clip_lower)
        
        forecasts = np.clip(forecasts, clip_lower, clip_upper)
    # If all context values are the same and non-negative (e.g., [5,5,5]), the forecast should ideally be 5.
    # The `np.maximum(0.0, forecasts)` above handles non-negativity.
    # The trend would be 0, so forecasts would be `level` (which is 5), correctly.
    
    return forecasts