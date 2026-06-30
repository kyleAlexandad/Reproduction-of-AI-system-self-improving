import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    """
    GIFT-Eval `m4_hourly/H/short` -- univariate time-series forecasting (freq "H").
    Forecasts the next `prediction_length` steps for a single hourly time series.

    This function implements a seasonal forecasting model with a daily seasonality
    (period 24). It averages the last K full days to get a robust seasonal pattern
    and applies a level adjustment based on the most recent full season's mean.
    It includes robust handling for short contexts and potential NaN/Inf values.

    Args:
        context (np.ndarray): 1D array of past target values (history only).
        prediction_length (int): The forecast horizon.
        freq (str): A frequency string, e.g., "H".
        metadata (dict, optional): Optional dict containing series metadata.

    Returns:
        np.ndarray: 1D array of length `prediction_length` with point forecasts.
    """
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- Robust Fallback Value for NaN/Inf Handling ---
    # Prioritize the last valid (non-NaN, non-Inf) value from context.
    # If no valid values exist, default to 0.0.
    fallback_for_nan_inf = 0.0
    if n > 0:
        valid_context = context[~np.isnan(context) & ~np.isinf(context)]
        if len(valid_context) > 0:
            fallback_for_nan_inf = float(valid_context[-1])
        # Else: all context values are NaN/Inf, fallback_for_nan_inf remains 0.0

    # Hourly data for M4 hourly series typically has strong daily seasonality (24 hours).
    season_period = 24 

    # --- Handle Very Short or Empty Contexts ---
    if n == 0:
        # If no history, return zeros or a sane default.
        return np.full(prediction_length, fallback_for_nan_inf, dtype=float)
    elif n < season_period:
        # If not enough data for a full season, fallback to last-value naive forecast.
        return np.full(prediction_length, fallback_for_nan_inf, dtype=float)

    # --- Core Seasonal Forecasting Logic ---

    # Determine K: number of full seasons (days) to average for the seasonal pattern.
    # We limit K to a reasonable maximum (e.g., 8 days) to avoid going too far back
    # and to keep computation light, while also ensuring at least 1 full season.
    K = min(max(1, n // season_period), 8)

    # Extract the last K full seasons from the context.
    # This slice is safe due to the checks n >= season_period and K >= 1.
    relevant_data = context[-K * season_period:]
    
    # Reshape the data into a (K, season_period) matrix, where each row represents a day.
    mat = relevant_data.reshape(K, season_period)

    # Calculate the average seasonal pattern (mean for each hour of the day) across K days.
    # Use np.nanmean to robustly handle any potential NaN values in the history data.
    seasonal_pattern = np.nanmean(mat, axis=0)

    # --- Level Adjustment ---
    # To improve upon a simple seasonal forecast, we apply a level adjustment.
    # This adjusts the overall forecast level based on the difference between
    # the mean of the most recent full season in context and the mean of the
    # derived seasonal pattern.

    # Mean of the most recent complete season in the context.
    # This is safe because n >= season_period.
    recent_level = np.nanmean(context[-season_period:])
    
    # Mean of the calculated seasonal pattern.
    seasonal_pattern_level = np.nanmean(seasonal_pattern)
    
    # Calculate the level offset. If any of the means are NaN, the offset will be NaN.
    level_offset = recent_level - seasonal_pattern_level

    # If the level_offset calculation resulted in NaN or Inf (e.g., if all relevant
    # data was NaN), reset it to 0.0 to prevent propagating bad values.
    if np.isnan(level_offset) or np.isinf(level_offset):
        level_offset = 0.0

    # --- Generate Forecast Output ---

    # Calculate the number of repetitions needed to cover the prediction_length.
    reps = int(np.ceil(prediction_length / season_period))
    
    # Tile the seasonal pattern to create a raw forecast.
    out = np.tile(seasonal_pattern, reps)
    
    # Trim the tiled pattern to the exact prediction_length.
    out = out[:prediction_length]

    # Apply the calculated level offset to the forecast.
    out += level_offset

    # --- Final Robustness Check: Handle NaN/Inf in Output ---
    # Ensure no NaN or Inf values are present in the final output.
    # Any remaining NaNs (e.g., if an entire hour across K days was NaN) will be
    # replaced by the robust `fallback_for_nan_inf`.
    return np.nan_to_num(out, nan=fallback_for_nan_inf,
                         posinf=fallback_for_nan_inf,
                         neginf=fallback_for_nan_inf).astype(float)