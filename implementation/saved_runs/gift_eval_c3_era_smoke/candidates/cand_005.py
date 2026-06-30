import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    """
    Performs univariate time-series forecasting on weekly data.
    This model combines robust fallbacks for very short series,
    Simple Exponential Smoothing (SES) for series shorter than a full season,
    and an additive seasonal-trend model with smoothed components for longer series.

    For longer series (at least one full season), it estimates:
    - A smoothed recent level from the end of the context.
    - An additive seasonal pattern derived from the last complete season.
    - A linear trend component, estimated either from the difference in means
      of the last two seasons or via linear regression over the last season.
    A damped version of this trend is then applied.

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
    
    # Determine the season length. Default to 52 for weekly data.
    # The M4 Weekly dataset typically has a season length of 52.
    season_length = metadata.get('season_length', 52) if metadata else 52 

    # --- 1. Handle very short context arrays robustly ---
    if len(context) == 0:
        # If no history, forecast zeros.
        return np.zeros(prediction_length, dtype=float)
    
    if len(context) == 1:
        # If only one data point, repeat it (last-value naive).
        return np.repeat(context[-1], prediction_length).astype(float)

    # --- 2. Strategy for contexts shorter than a full season ---
    # Not enough data to reliably observe seasonality. Use Simple Exponential Smoothing (SES).
    # This provides a smoothed level estimate, which is more robust than just the last value.
    if len(context) < season_length:
        alpha = 0.2  # Smoothing parameter for SES.
        
        # Initialize smoothed level with the first observed value
        smoothed_level = context[0]
        # Iterate through the context to apply exponential smoothing
        for i in range(1, len(context)):
            smoothed_level = alpha * context[i] + (1 - alpha) * smoothed_level
            
        # Forecasts are the last computed smoothed level repeated.
        forecasts = np.repeat(smoothed_level, prediction_length).astype(float)
        
        # Ensure forecasts are non-negative, as is typical for many time series.
        return np.maximum(0, forecasts) 

    # --- 3. Strategy for contexts long enough to observe seasonality (len(context) >= season_length) ---
    # This approach combines a recent smoothed level, an additive seasonal component, and a damped linear trend.

    # 3.1. Estimate a recent smoothed level
    # Use the mean of the last few observed points to get a stable recent level.
    # Using min(len(context), 4) ensures we don't try to average more points than available.
    recent_level = np.mean(context[-min(len(context), 4):])

    # 3.2. Extract additive seasonal pattern
    # Use the last complete season's data to derive seasonal deviations around its mean.
    # This provides a seasonal pattern centered around zero, making it additive.
    last_season_data = context[-season_length:]
    last_season_mean = np.mean(last_season_data)
    seasonal_deviations = last_season_data - last_season_mean

    # 3.3. Estimate trend
    # Robust trend calculation:
    # If at least two full seasons are available, calculate trend from the difference
    # in means of the last two seasons. This makes the trend estimate more stable
    # for seasonal data.
    # Otherwise, if at least one full season is available, use linear regression
    # over the last season's data to get a local trend.
    trend_rate = 0.0
    if len(context) >= 2 * season_length:
        mean_last_season = np.mean(context[-season_length:])
        mean_prev_season = np.mean(context[-2*season_length:-season_length])
        trend_rate = (mean_last_season - mean_prev_season) / season_length
    else: # season_length <= len(context) < 2 * season_length
        # Fit a linear trend over the available full season.
        # np.polyfit requires at least 2 points for a degree 1 polynomial.
        # For weekly data, season_length is 52, so this condition is always met.
        if season_length >= 2:
            x_vals = np.arange(season_length)
            y_vals = context[-season_length:]
            # polyfit returns [slope, intercept] for degree 1
            slope, _ = np.polyfit(x_vals, y_vals, 1)
            trend_rate = slope
        # If season_length is 1, it implies no seasonality, and this case would
        # likely have been handled by the SES block if len(context) < 1.
        # But for robustness, if SL=1 and this branch is hit, trend_rate remains 0.0.

    # Damping factor for the trend. Reduces the impact of the trend as forecasts extend further.
    # A value of 0.9 is commonly used. The (h+1) exponent matches the parent's logic,
    # applying damping starting from the very first forecast step.
    phi = 0.9 

    forecasts = np.zeros(prediction_length, dtype=float)
    for h in range(prediction_length):
        # Forecast formula: recent_level + seasonal_effect + damped_trend
        # The seasonal effect is picked from the derived seasonal pattern.
        seasonal_effect = seasonal_deviations[h % season_length]
        
        # The trend component grows linearly with the forecast horizon `h`,
        # and its influence is progressively reduced by the damping factor `phi`.
        damped_trend_component = trend_rate * (h + 1) * (phi**(h + 1))
        
        forecasts[h] = recent_level + seasonal_effect + damped_trend_component

    # Ensure all forecasts are non-negative, as is typical for many time series.
    return np.maximum(0, forecasts)