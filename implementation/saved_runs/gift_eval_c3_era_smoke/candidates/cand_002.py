import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)

    # 1. Handle very short context arrays robustly
    if len(context) == 0:
        return np.zeros(prediction_length)
    if len(context) == 1:
        # For a single point, forecast it repeatedly (last-value naive)
        return np.repeat(context[0], prediction_length)

    # Define parameters for weekly seasonality and smoothing
    season_length = 52  # M4 Weekly data typically has a 52-week (yearly) seasonality
    alpha = 0.3         # Smoothing parameter for Simple Exponential Smoothing (SES)
    
    # Damping factors for trend to prevent runaway forecasts
    trend_damping_factor_seasonal = 0.7 
    trend_damping_factor_non_seasonal = 0.5 

    forecasts = np.zeros(prediction_length)

    # 2. Calculate Simple Exponential Smoothing (SES) level
    # Initialize the smoothed level with the first observed value
    level = context[0]
    # Update the level using all historical data
    # Iteratively calculate the smoothed value S_t = alpha * y_t + (1 - alpha) * S_{t-1}
    for i in range(1, len(context)):
        level = alpha * context[i] + (1 - alpha) * level

    # 3. Calculate trend component
    trend = 0.0
    
    # Determine the window size for trend calculation
    # Use a longer window (e.g., half a season) if enough data for seasonality
    # Otherwise, use a shorter window (e.g., up to 10 points)
    if len(context) >= season_length:
        trend_window = min(len(context), season_length // 2) 
    else:
        trend_window = min(len(context), 10) 

    if trend_window > 1:
        # Extract recent history for trend calculation
        recent_history = context[-trend_window:]
        # Simple linear trend: difference between last and first point in the window
        trend = (recent_history[-1] - recent_history[0]) / (trend_window - 1)
        
        # Apply damping to the calculated trend
        if len(context) >= season_length:
            trend *= trend_damping_factor_seasonal
        else:
            trend *= trend_damping_factor_non_seasonal

    # 4. Generate forecasts based on available history for seasonality
    if len(context) >= season_length:
        # Case: Enough history for seasonality (using multiplicative seasonality)

        # Extract the last full season's data to determine the seasonal pattern
        seasonal_pattern = context[-season_length:]
        seasonal_average = np.mean(seasonal_pattern)

        # Calculate seasonal factors (ratio of actual values to the seasonal average)
        if seasonal_average != 0:
            seasonal_factors = seasonal_pattern / seasonal_average
        else:
            # Fallback: if the seasonal average is zero (e.g., all zeros in the last season),
            # assume no seasonality (factors are all 1) to avoid division by zero and propagate zeros
            seasonal_factors = np.ones_like(seasonal_pattern)

        for h in range(prediction_length):
            # Base forecast: smoothed level extrapolated with trend
            base_forecast = level + trend * (h + 1)
            
            # Apply the multiplicative seasonal factor for the corresponding period
            # Using modulo operator makes it robust if prediction_length > season_length
            seasonal_factor_for_h = seasonal_factors[h % season_length]
            forecasts[h] = base_forecast * seasonal_factor_for_h
    else:
        # Case: Not enough history for seasonality (SES + Trend only)
        # Forecast is based solely on the smoothed level and trend extrapolation
        for h in range(prediction_length):
            forecasts[h] = level + trend * (h + 1)

    # 5. Clip forecasts to prevent negative values, as M4 series often represent non-negative quantities (e.g., sales)
    return np.maximum(0., forecasts)