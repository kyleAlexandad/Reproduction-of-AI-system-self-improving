import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    """
    Forecasts future values of a univariate time series using a seasonal averaging
    method with a level adjustment. Designed for hourly data with strong daily seasonality.

    Args:
        context (np.ndarray): A 1D numpy array of past target values (history only).
        prediction_length (int): The number of future steps to forecast (horizon).
        freq (str): The frequency string (e.g., "H" for hourly).
        metadata (dict, optional): Optional metadata (e.g., item_id, season_length).

    Returns:
        np.ndarray: A 1D numpy array of length `prediction_length` with point forecasts.
    """
    context = np.asarray(context, dtype=float)
    n = len(context)
    
    # Define the primary seasonal period. For hourly M4 data, daily seasonality (24 hours)
    # is known to be very strong and critical for good performance.
    season_period = 24  
    
    # --- Handle extremely short contexts ---
    # If there's no history, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # If context is shorter than one full season, we cannot reliably extract a seasonal pattern.
    # In this case, fall back to a simple last-value naive forecast.
    if n < season_period:
        return np.full(prediction_length, context[-1], dtype=float)

    # --- Core Seasonal Averaging Logic ---
    
    # Determine how many full seasons (days) to average.
    # We ensure at least 1 full season is used (since n >= season_period is guaranteed here).
    # We cap the number of seasons to average at 8 to prevent using too old data,
    # and to keep the computation lightweight and responsive to recent changes.
    # The number of seasons must also not exceed what's available in the context.
    num_seasons_to_average = min(max(1, n // season_period), 8)
    
    # Extract the most recent `num_seasons_to_average` full seasons from the context.
    # This slicing is safe because `num_seasons_to_average * season_period` is guaranteed
    # to be less than or equal to `n`.
    recent_seasons_data = context[-num_seasons_to_average * season_period:]
    
    # Reshape the data into a matrix: each row represents a full season (day),
    # and each column represents a specific hour within the day.
    seasonal_matrix = recent_seasons_data.reshape(num_seasons_to_average, season_period)
    
    # Calculate the average value for each hour of the day across the `K` recent seasons.
    # Using `np.nanmean` makes this robust to potential NaN values in the input `context`.
    seasonal_pattern = np.nanmean(seasonal_matrix, axis=0)
    
    # --- Optional: Add a simple level adjustment ---
    # This step attempts to align the overall level of the forecast with the most recent observed data.
    
    # Calculate the mean of the most recent *full* season available in the `context`.
    # `np.nanmean` is used for robustness against NaNs.
    mean_last_observed_season = np.nanmean(context[-season_period:])
    
    # Calculate the mean of our derived seasonal pattern.
    mean_seasonal_pattern = np.nanmean(seasonal_pattern)
    
    # Initialize the level shift to 0.0.
    level_shift = 0.0
    
    # Apply a level shift only if both means are finite (not NaN or Inf).
    # This prevents NaN propagation if, for example, the last observed season
    # was entirely composed of NaNs, making `mean_last_observed_season` a NaN.
    if np.isfinite(mean_last_observed_season) and np.isfinite(mean_seasonal_pattern):
        level_shift = mean_last_observed_season - mean_seasonal_pattern
    
    # Apply the calculated level shift to the seasonal pattern.
    # If `seasonal_pattern` itself contains NaNs (because all values for a given hour were NaN),
    # these NaNs will persist in `adjusted_seasonal_pattern`. They will be handled by `nan_to_num` later.
    adjusted_seasonal_pattern = seasonal_pattern + level_shift
    
    # --- Generate the final forecast ---
    
    # Calculate how many full repetitions of the `adjusted_seasonal_pattern` are needed
    # to cover the entire `prediction_length`.
    repetitions = int(np.ceil(prediction_length / season_period))
    
    # Tile the adjusted seasonal pattern and then slice it to ensure the output
    # has exactly `prediction_length` values.
    raw_forecast = np.tile(adjusted_seasonal_pattern, repetitions)[:prediction_length]
    
    # Ensure no NaN or Inf values in the final output.
    # Replace any non-finite values (NaN, Inf) with the last known valid context value (`context[-1]`).
    # This handles any NaNs that might have propagated from `seasonal_pattern` or `level_shift` calculations.
    final_forecast = np.nan_to_num(raw_forecast, 
                                   nan=float(context[-1]), 
                                   posinf=float(context[-1]), 
                                   neginf=float(context[-1]))
    
    # Ensure the final output array is of float type.
    return final_forecast.astype(float)
