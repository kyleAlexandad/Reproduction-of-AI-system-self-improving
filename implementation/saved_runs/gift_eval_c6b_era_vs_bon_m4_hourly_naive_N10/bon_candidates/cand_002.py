import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    """
    Univariate time-series forecasting function for hourly data, exploiting daily and weekly seasonality.

    Args:
        context (np.ndarray): A 1D numpy array of the past target values for one series.
        prediction_length (int): The forecast horizon.
        freq (str): A frequency string, e.g., "H".
        metadata (dict, optional): An optional dictionary with series metadata. Defaults to None.

    Returns:
        np.ndarray: A 1D numpy array of length `prediction_length` with the point forecasts.
    """
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- Hard Requirement: Handle empty context ---
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Define seasonal periods based on hourly data and problem statement
    season_daily = 24  # Daily seasonality (24 hours)
    season_weekly = 168 # Weekly seasonality (7 days * 24 hours/day)

    # --- Daily seasonality component ---
    # Fallback for contexts shorter than a full daily season
    if n < season_daily:
        # Hard Requirement: Fallback to last-value naive forecast for short contexts.
        # Ensure context[-1] is not NaN; if it is, nan_to_num will handle it later.
        last_val = float(context[-1]) if n > 0 else 0.0 # Guard against n=0 (already handled, but defensive)
        return np.full(prediction_length, last_val)

    # Calculate K_daily: number of full daily seasons to average.
    # Max of 1 to ensure at least one full season is used if available.
    # Capped at 8 to avoid using overly stale data from too far in the past.
    K_daily = min(max(1, n // season_daily), 8)
    
    # Extract the most recent K_daily * season_daily points from the context.
    # This slice is guaranteed to have enough points because n >= season_daily and K_daily >= 1.
    daily_data_for_pattern = context[-K_daily * season_daily:]
    daily_mat = daily_data_for_pattern.reshape(K_daily, season_daily)
    
    # Calculate the mean for each hour of the day across the K_daily seasons.
    # np.nanmean ignores NaN values in the computation. If all values for a specific hour are NaN,
    # the result for that hour will be NaN, which is handled by nan_to_num at the end.
    daily_seasonal_pattern_base = np.nanmean(daily_mat, axis=0)

    # Determine the starting index for the forecast from the daily seasonal pattern.
    # If the context ends at index (n-1), the first forecast point (out[0]) corresponds to index n.
    # The position in the daily cycle for index n is n % season_daily.
    start_idx_daily = n % season_daily
    
    # Roll the base seasonal pattern so that its first element is correctly aligned with `start_idx_daily`.
    # This ensures the forecast starts at the correct point in the daily cycle.
    daily_seasonal_pattern_aligned = np.roll(daily_seasonal_pattern_base, -start_idx_daily)
    
    # Tile the aligned daily pattern to cover the entire prediction length.
    reps_daily = int(np.ceil(prediction_length / season_daily))
    daily_forecast_component = np.tile(daily_seasonal_pattern_aligned, reps_daily)[:prediction_length]

    # --- Weekly seasonality component (if enough data) ---
    # If there isn't enough data for a full weekly season, we rely solely on the daily component.
    if n < season_weekly:
        # Hard Requirement: Ensure no NaN/inf in output.
        # Fallback for NaNs in the daily forecast to the last observed context value.
        last_val = float(context[-1]) if n > 0 else 0.0
        return np.nan_to_num(daily_forecast_component, nan=last_val).astype(float)

    # Calculate K_weekly: number of full weekly seasons to average.
    # Capped at 4 to use up to approximately the last month of data, balancing recency and smoothing.
    K_weekly = min(max(1, n // season_weekly), 4)
    
    # Extract the most recent K_weekly * season_weekly points.
    weekly_data_for_pattern = context[-K_weekly * season_weekly:]
    weekly_mat = weekly_data_for_pattern.reshape(K_weekly, season_weekly)
    
    # Calculate the mean for each hour of the week, ignoring NaN values.
    weekly_seasonal_pattern_base = np.nanmean(weekly_mat, axis=0)

    # Determine the starting index for the forecast from the weekly seasonal pattern.
    start_idx_weekly = n % season_weekly
    weekly_seasonal_pattern_aligned = np.roll(weekly_seasonal_pattern_base, -start_idx_weekly)
    
    # Tile the aligned weekly pattern.
    reps_weekly = int(np.ceil(prediction_length / season_weekly))
    weekly_forecast_component = np.tile(weekly_seasonal_pattern_aligned, reps_weekly)[:prediction_length]

    # --- Blending the daily and weekly seasonal components ---
    # The problem statement notes daily seasonality is "STRONG" and weekly "may also help".
    # A higher weight is given to the daily component. This factor can be tuned.
    alpha_daily = 0.7 
    blended_forecast = alpha_daily * daily_forecast_component + (1 - alpha_daily) * weekly_forecast_component

    # --- Final Robustness: Replace any NaN/inf in the final forecast ---
    # This catches NaNs from np.nanmean if all values for a specific hour were NaN,
    # or any other potential numerical issues.
    # The replacement value is the last observed value in the context.
    last_val = float(context[-1]) if n > 0 else 0.0
    return np.nan_to_num(blended_forecast, nan=last_val, posinf=last_val, neginf=last_val).astype(float)
