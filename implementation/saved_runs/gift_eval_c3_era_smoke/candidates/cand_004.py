import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    """
    Performs univariate time-series forecasting using a hybrid model that
    combines Simple Exponential Smoothing (SES) for short series and a
    seasonal-trend model for longer series.

    For short series (less than a full season), it uses SES to estimate a robust
    level and repeats it.
    For longer series (at least one full season), it forecasts by taking
    the seasonal component from the last observed cycle, adjusting it for
    the current overall level, and adding a damped linear trend.

    Args:
        context (np.ndarray): 1D numpy array of past target values (history only).
        prediction_length (int): The forecast horizon.
        freq (str): A frequency string, e.g., "W" for weekly.
        metadata (dict, optional): An optional dictionary that may contain
                                   keys like 'item_id', 'season_length'.

    Returns:
        np.ndarray: A 1D numpy array of length `prediction_length` with point forecasts.
    """
    context = np.asarray(context, dtype=float)
    
    # Determine the season length. Default to 52 for weekly data if not provided in metadata.
    season_length = metadata.get('season_length', 52) if metadata else 52 

    # --- 1. Handle very short context arrays robustly ---
    if len(context) == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # If only one data point, repeat it (last-value naive for minimal history)
    if len(context) == 1:
        return np.repeat(context[-1], prediction_length).astype(float)

    # --- 2. Strategy for contexts shorter than a full season ---
    # Not enough data to reliably observe seasonality. Use Simple Exponential Smoothing (SES).
    # SES provides a smoothed level estimate, which is more robust than just the last value.
    if len(context) < season_length:
        alpha = 0.2  # Smoothing parameter for SES. Can be tuned, but fixed for robustness.
        
        # Initialize the smoothed level with the first observed value
        smoothed_level = context[0]
        # Iterate through the context to apply exponential smoothing
        for i in range(1, len(context)):
            smoothed_level = alpha * context[i] + (1 - alpha) * smoothed_level
            
        # Forecasts are just the last computed smoothed level repeated for the prediction_length
        forecasts = np.repeat(smoothed_level, prediction_length).astype(float)
        
        # Ensure forecasts are non-negative, as time series often represent positive quantities
        return np.maximum(0, forecasts) 

    # --- 3. Strategy for contexts long enough to observe seasonality (len(context) >= season_length) ---
    # This approach combines a seasonal component, an adjusted level, and a damped linear trend.

    # Current Level: The most recent observation is a good proxy for the current level of the series.
    current_level = context[-1]

    # Level from one season ago: This value is crucial for calculating the seasonal drift/trend.
    prev_seasonal_level = context[-season_length]
    
    # Trend Component: Estimate the trend based on the drift observed over the last full season.
    # This calculates the average change per period (week) over the last `season_length` periods.
    trend_per_period = (current_level - prev_seasonal_level) / season_length

    # Damping factor for the trend. Reduces the impact of the trend as forecasts extend further
    # into the future, preventing runaway forecasts. A value of 0.9 is commonly used.
    phi = 0.9 

    forecasts = np.zeros(prediction_length, dtype=float)
    for h in range(prediction_length):
        # Determine the seasonal component by looking at the value from the last observed cycle.
        # `(h % season_length)` ensures we correctly cycle through the seasonal pattern if
        # `prediction_length` is greater than `season_length`.
        seasonal_idx_in_context = len(context) - season_length + (h % season_length)
        seasonal_value_from_past = context[seasonal_idx_in_context]

        # Adjust the seasonal value to the current level of the series.
        # The `seasonal_value_from_past` is `season_length` periods old. We need to shift its
        # base level to match the `current_level` relative to `prev_seasonal_level`.
        # This implicitly incorporates the overall level shift observed in the last season.
        adjusted_seasonal_value = seasonal_value_from_past - prev_seasonal_level + current_level

        # Add the damped trend component.
        # The trend is applied incrementally for each forecast step `(h + 1)`.
        # The damping factor `phi` is raised to `(h + 1)` to progressively reduce the trend's influence.
        damped_trend_component = trend_per_period * (h + 1) * (phi**(h + 1))
        
        forecasts[h] = adjusted_seasonal_value + damped_trend_component

    # Ensure all forecasts are non-negative, as is typical for many time series.
    return np.maximum(0, forecasts)
