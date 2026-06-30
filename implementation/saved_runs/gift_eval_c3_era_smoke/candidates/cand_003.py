import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    """
    Forecasts future values of a univariate time series using a combination of
    Damped Linear Trend (DLT) and a blended Seasonal Naive (SN) component.

    Handles short series robustly and applies non-negativity clipping.

    Args:
        context (np.array): A 1D numpy array of past target values (history only).
        prediction_length (int): The forecast horizon.
        freq (str): Frequency string, e.g., "W" for weekly.
        metadata (dict, optional): Optional dict with additional info like 'season_length'.

    Returns:
        np.array: A 1D numpy array of length `prediction_length` with point forecasts.
    """
    context = np.asarray(context, dtype=float)

    # --- Configuration Parameters ---
    # Seasonality for weekly data (52 weeks in a year). Can be overridden by metadata.
    season_length = metadata.get('season_length', 52) if metadata else 52

    # Span for calculating the recent level (simple moving average)
    smoothing_span = 7

    # Window for calculating the recent trend (average difference over last N points)
    trend_window = 4

    # Damping factor for trend extrapolation (0.0 to 1.0, closer to 1.0 means less damping)
    # A value of 0.95 often works well to prevent runaway trends.
    damping_factor = 0.95

    # Weight for blending the seasonal component (0.0 for pure DLT, 1.0 for pure Seasonal Naive)
    # Applied only if sufficient seasonal history is available.
    seasonal_blend_weight = 0.5

    # Minimum number of full seasonal cycles required to enable the seasonal component.
    # Set to 1 to allow seasonal influence even with just one full year of history.
    min_seasonal_cycles_for_blend = 1

    # --- Fallbacks for very short series ---
    if len(context) == 0:
        return np.zeros(prediction_length, dtype=float)
    if len(context) == 1:
        # If only one historical point, repeat it for the entire forecast horizon.
        return np.repeat(context[0], prediction_length).astype(float)

    # --- Calculate Base Level ---
    # The current level is estimated as the mean of the most recent points.
    # The span is capped by the available context length.
    current_level_span = min(smoothing_span, len(context))
    current_level = np.mean(context[-current_level_span:])

    # --- Calculate Trend ---
    current_trend = 0.0
    # Trend requires at least 2 points to calculate a difference.
    if len(context) >= 2:
        trend_slice_len = min(trend_window, len(context))
        if trend_slice_len >= 2:
            # Trend is calculated as the average difference between the first and last point
            # in the trend window, effectively a robust linear slope.
            current_trend = (context[-1] - context[-trend_slice_len]) / (trend_slice_len - 1)

    # --- Generate Damped Linear Trend (DLT) Forecast ---
    dlt_forecasts = np.zeros(prediction_length, dtype=float)
    temp_level = current_level
    temp_trend = current_trend

    for h in range(prediction_length):
        # Forecast for step h is the current level plus the current (damped) trend.
        dlt_forecasts[h] = temp_level + temp_trend
        
        # Update level for the *next* forecast step.
        temp_level += temp_trend
        
        # Apply damping to the trend for the next step.
        temp_trend *= damping_factor

    # --- Incorporate Seasonality (blended with DLT if enough history) ---
    final_forecasts = dlt_forecasts.copy()

    # Seasonality is considered only if there's enough history for at least one full cycle.
    if len(context) >= season_length * min_seasonal_cycles_for_blend:
        
        # For each forecast step `h`, we find the corresponding historical point `season_length` weeks ago.
        # This provides a 'seasonal naive' component based on the last observed value for that season.
        for h in range(prediction_length):
            # Calculate the index of the corresponding point in history for seasonal comparison.
            # E.g., for h=0, we look at `context[len(context) - season_length]`.
            # For h=1, we look at `context[len(context) - season_length + 1]`.
            seasonal_naive_idx_in_history = len(context) - season_length + h
            
            if seasonal_naive_idx_in_history >= 0:
                # If there's a historical value for this specific seasonal point, use it.
                seasonal_val_from_history = context[seasonal_naive_idx_in_history]
                
                # Blend the DLT forecast with the seasonal naive value.
                final_forecasts[h] = (1 - seasonal_blend_weight) * dlt_forecasts[h] + \
                                      seasonal_blend_weight * seasonal_val_from_history
            # Else, if not enough history for this specific seasonal point (e.g., first few weeks of
            # the forecast horizon might precede the available seasonal history), we rely solely on DLT.
            # This is already covered by `final_forecasts = dlt_forecasts.copy()` and not overriding.

    # --- Post-processing: Clipping ---
    # Ensure forecasts are non-negative, as many real-world series (like weekly sales) are.
    # This helps in preventing unrealistic negative forecasts.
    final_forecasts[final_forecasts < 0] = 0

    # Handle the specific edge case where the entire context is zeros.
    # In this scenario, the forecast should also be all zeros.
    if np.all(context == 0):
        return np.zeros(prediction_length, dtype=float)

    return final_forecasts