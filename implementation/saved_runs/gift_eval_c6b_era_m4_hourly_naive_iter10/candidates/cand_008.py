import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    # If no historical data is available, return zeros for the forecast.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: Hourly data often exhibits strong daily seasonality (period 24).
    # This is the primary seasonality period to leverage.
    season = 24

    # Hard requirement 8: Fallback for contexts shorter than one full season.
    # In this case, use the last observed value as a naive forecast for all steps.
    # context[-1] is safe because n > 0.
    if n < season:
        return np.full(prediction_length, float(context[-1]))

    # --- Step 1: Establish a robust seasonal pattern by averaging recent cycles ---
    # The prompt suggests "average the last K full days at each hour to denoise the pattern".
    # This is an improvement over simply taking the last single season, as done by the parent candidate.
    
    # Determine the number of complete seasonal cycles (days) available in the context.
    # We cap this at 8, as suggested, to avoid using excessively old data which might not be relevant.
    # We ensure K is at least 1, as n >= season has been checked.
    K_seasons_to_average = min(max(1, n // season), 8)
    
    # Extract the data corresponding to these last K complete seasonal cycles.
    # Indexing safety: `n // season` ensures that `K_seasons_to_average * season` is
    # less than or equal to `n`, so `context[-K_seasons_to_average * season:]` is always safe.
    seasonal_data_for_averaging = context[-K_seasons_to_average * season:]
    
    # Reshape the data into (number_of_cycles, season_length) and compute the mean
    # across the cycles to get a robust hourly average for the day.
    averaged_seasonal_pattern = seasonal_data_for_averaging.reshape(K_seasons_to_average, season).mean(axis=0)

    # Tile this averaged seasonal pattern to cover the entire prediction length.
    reps = int(np.ceil(prediction_length / season))
    base_seasonal_forecasts = np.tile(averaged_seasonal_pattern, reps)[:prediction_length]

    # Initialize the final forecasts array with the base seasonal pattern.
    forecasts = base_seasonal_forecasts.copy()

    # --- Step 2: Add a damped level/trend correction ---
    # This step incorporates a short-term trend on top of the seasonal pattern.
    # It requires at least two full seasonal cycles to estimate a meaningful trend
    # by comparing the most recent observation to its seasonal counterpart from the previous cycle.
    if n >= 2 * season:
        # Calculate the recent level change over one seasonal period.
        # This is `context[t] - context[t - season]`, capturing a deviation from the
        # seasonal expectation at the very end of the context.
        # Indexing safety: n >= 2 * season guarantees both context[-1] and context[-1 - season] are valid.
        level_trend_over_season = context[-1] - context[-1 - season]

        # Apply exponential damping to the trend. A factor of 0.95 (as in the parent candidate)
        # ensures the trend's influence diminishes quickly into the future, preventing over-extrapolation.
        damping_factor = 0.95

        # Apply the damped trend to each forecast step.
        # The exponent `(h + 1)` correctly applies the damping effect for `h` steps into the future
        # (h=0 is the first forecast step, so it's 1 step ahead).
        for h in range(prediction_length):
            forecasts[h] += level_trend_over_season * (damping_factor ** (h + 1))

    # Hard requirement 4: Never output NaN or inf. Clip/guard as needed.
    # `np.nan_to_num` replaces any potential NaNs (e.g., if input context itself contained NaNs)
    # with a specified fallback value. Using the last valid context value is a reasonable default.
    # If n=0, context[-1] would fail, but that's handled by the `if n == 0` block.
    fallback_value_for_nan = float(context[-1]) if n > 0 else 0.0
    
    return np.nan_to_num(forecasts, nan=fallback_value_for_nan).astype(float)