import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    # If no historical data is available, return zeros for the forecast.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # --- Robust Last Known Value for Fallbacks ---
    # Find the last non-NaN value in the context. If all are NaN, default to 0.0.
    last_known_value = 0.0
    # Create a boolean mask for non-NaN values
    non_nan_mask = ~np.isnan(context)
    if np.any(non_nan_mask):
        # Get the last non-NaN value
        last_known_value = float(context[non_nan_mask][-1])
    # If all context values are NaN, last_known_value remains 0.0, which is a safe fallback.
    
    # Critical benchmark fact: Hourly data often exhibits strong daily seasonality (period 24).
    season = 24

    # Hard requirement 8: Fallback for contexts shorter than one full season.
    # In this case, use the robust last known value as a naive forecast for all steps.
    # This also handles the case where context contains NaNs, by using last_known_value.
    if n < season:
        return np.full(prediction_length, last_known_value)

    # --- Step 1: Establish a robust seasonal pattern by averaging recent cycles ---
    # Determine the number of complete seasonal cycles (days) available in the context.
    # We cap this at 8 to avoid using excessively old data, as suggested by common practice.
    # `n // season` gives the number of full seasons available. `max(1, ...)` ensures K is at least 1.
    K_seasons_to_average = min(max(1, n // season), 8)
    
    # Extract the data corresponding to these last K complete seasonal cycles.
    # Indexing safety: `K_seasons_to_average * season` is less than or equal to `n`,
    # so `context[-K_seasons_to_average * season:]` is always a valid slice.
    seasonal_data_for_averaging = context[-K_seasons_to_average * season:]
    
    # Reshape the data into (number_of_cycles, season_length) and compute the mean
    # across the cycles to get a robust hourly average for the day.
    # `np.nanmean` is used to handle potential NaNs within the historical data
    # gracefully; if a specific hour across K cycles is all NaN, its average will be NaN.
    averaged_seasonal_pattern = np.nanmean(seasonal_data_for_averaging.reshape(K_seasons_to_average, season), axis=0)

    # Tile this averaged seasonal pattern to cover the entire prediction length.
    reps = int(np.ceil(prediction_length / season))
    # Initialize the forecasts array with this base seasonal pattern.
    forecasts = np.tile(averaged_seasonal_pattern, reps)[:prediction_length]

    # --- Step 2: Add a damped level adjustment/trend correction ---
    # This step incorporates a short-term level adjustment based on the deviation
    # of the last observed value from its seasonal expectation.
    # Apply only if sufficient history (at least two full seasons) is available.
    # This provides a more stable seasonal pattern for comparison.
    if n >= 2 * season:
        # Get the seasonal index for the last observed point's hour (0-indexed).
        last_seasonal_index = (n - 1) % season
        
        # Get the averaged seasonal expectation for that specific hour.
        seasonal_expectation_at_last_point = averaged_seasonal_pattern[last_seasonal_index]

        # Calculate the deviation of the last actual observation from its seasonal expectation.
        # This acts as the "current level shift" or residual to be damped forward.
        current_level_deviation = 0.0
        # Only calculate if the last observation itself is not NaN and its seasonal expectation is not NaN.
        if not np.isnan(context[-1]) and not np.isnan(seasonal_expectation_at_last_point):
             current_level_deviation = context[-1] - seasonal_expectation_at_last_point
        # If either is NaN, current_level_deviation remains 0.0, effectively disabling the trend.
        
        # Apply exponential damping to this level deviation. A factor of 0.95 is typical.
        damping_factor = 0.95 

        # Add the damped deviation to each forecast step.
        # `h + 1` correctly applies the damping for `h` steps into the future (h=0 is 1 step ahead).
        for h in range(prediction_length):
            forecasts[h] += current_level_deviation * (damping_factor ** (h + 1))

    # Hard requirement 4: Never output NaN or inf. Clip/guard as needed.
    # `np.nan_to_num` replaces any potential NaNs (e.g., if the input context itself
    # contained NaNs that propagated to `averaged_seasonal_pattern`).
    # Use the robust `last_known_value` (calculated at the beginning) as the replacement.
    return np.nan_to_num(forecasts, nan=last_known_value).astype(float)