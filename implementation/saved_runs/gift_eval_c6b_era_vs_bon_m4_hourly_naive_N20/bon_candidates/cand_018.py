import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    """
    Univariate time-series forecasting function for hourly data with strong daily seasonality.

    This function implements a seasonal-average approach with a level adjustment.
    It leverages the strong daily seasonality (period 24) identified in the problem
    description for M4 hourly data. It is robust to short context arrays and
    ensures no NaNs or infs are returned.

    Args:
        context (np.ndarray): A 1D numpy array of past target values (history only).
        prediction_length (int): The forecast horizon.
        freq (str): A frequency string, e.g., "H". (Not directly used, but part of signature).
        metadata (dict, optional): An optional dictionary with series metadata. (Not directly used, but part of signature).

    Returns:
        np.ndarray: A 1D numpy array of length `prediction_length` with point forecasts.
    """
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- Robust fallback value determination ---
    # This value will be used if context[-1] is NaN, or if computations result in NaN.
    # It prioritizes: last non-NaN value in context, then 0.0 if all context is NaN or empty.
    fallback_val = 0.0
    if n > 0:
        if not np.isnan(context[-1]):
            fallback_val = float(context[-1])
        else:
            # Find the last non-NaN value in context
            last_valid_idx = np.where(~np.isnan(context))[0]
            if len(last_valid_idx) > 0:
                fallback_val = float(context[last_valid_idx[-1]])
            # else: all context values are NaN, fallback_val remains 0.0

    # Hard requirement 8: Handle short context arrays (length 0 or very few points)
    if n == 0:
        # If context is empty, return an array filled with the fallback_val (0.0).
        return np.full(prediction_length, fallback_val, dtype=float)

    season = 24  # Daily seasonality for hourly data (Critical Benchmark Fact)

    # Fallback for contexts shorter than one full season
    if n < season:
        # Hard requirement 8: Sane fallbacks
        # Hard requirement 4: Never output NaN or inf
        return np.full(prediction_length, fallback_val, dtype=float)

    # Calculate seasonal component by averaging the last K full days.
    # K is chosen to be between 1 and 8 full seasons, ensuring we have enough data
    # but not too old data for the seasonal pattern.
    K = min(max(1, n // season), 8)
    
    # Hard requirement 9: INDEXING SAFETY. Slicing from the end `context[-K * season:]`
    # is safe because `K * season` is guaranteed to be less than or equal to `n`.
    seasonal_data_segment = context[-K * season:]
    
    # Reshape into K rows, `season` columns. Each row represents a day's pattern.
    # np.reshape handles NaNs by carrying them over.
    mat = seasonal_data_segment.reshape(K, season)
    
    # Average across rows to get the seasonal pattern. `np.nanmean` robustly handles
    # potential NaNs within the `mat` (if present in `context`).
    seasonal_pattern = np.nanmean(mat, axis=0)
    
    # If `seasonal_pattern` itself became all NaNs (e.g., if the `K` seasons were all NaNs),
    # replace with `fallback_val` to prevent further NaN propagation.
    seasonal_pattern = np.nan_to_num(seasonal_pattern, nan=fallback_val)

    # --- Add a level adjustment component ---
    # This adjusts the seasonal forecast based on the most recent deviation from
    # the expected seasonal value. It assumes the current level shift persists.
    
    # Determine the seasonal index for the last observed point in `context`.
    last_point_seasonal_index = (n - 1) % season
    
    # Get the current value, using `fallback_val` if `context[-1]` is NaN.
    current_value = float(context[-1]) if not np.isnan(context[-1]) else fallback_val
    
    # Get the seasonal expectation for the last point's hour.
    # `seasonal_pattern` is guaranteed not to contain NaNs at this stage.
    seasonal_expectation_at_last_point = seasonal_pattern[last_point_seasonal_index]
    
    # Calculate the level shift: difference between current value and its seasonal expectation.
    level_shift = current_value - seasonal_expectation_at_last_point

    # Tile the seasonal pattern to cover the entire prediction length.
    reps = int(np.ceil(prediction_length / season))
    tiled_seasonal_forecast = np.tile(seasonal_pattern, reps)[:prediction_length]

    # Add the calculated level shift to the seasonal forecast.
    forecasts = tiled_seasonal_forecast + level_shift

    # Hard requirement 4: Never output NaN or inf.
    # Replace any remaining NaNs (if any due to unforeseen calculation paths) with `fallback_val`.
    final_forecast = np.nan_to_num(forecasts, nan=fallback_val)

    # Hard requirement 3: Output length must be EXACTLY prediction_length.
    # This is ensured by the `[:prediction_length]` slice earlier and the initial checks.

    return final_forecast.astype(float)