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
    non_nan_mask = ~np.isnan(context)
    if np.any(non_nan_mask):
        last_known_value = float(context[non_nan_mask][-1])
    # If all context values are NaN, last_known_value remains 0.0.

    # Critical benchmark fact: Hourly data often exhibits strong daily seasonality (period 24).
    season = 24

    # Hard requirement 8: Fallback for contexts shorter than one full season.
    # In this case, use the robust last known value as a naive forecast for all steps.
    if n < season:
        return np.full(prediction_length, last_known_value)

    # --- Establish a robust seasonal pattern by averaging recent cycles ---
    # Determine the number of complete seasonal cycles (days) available in the context.
    # We cap this at 8 to avoid using excessively old data, as suggested by common practice.
    K_seasons_to_average = min(max(1, n // season), 8)
    
    # Extract the data corresponding to these last K complete seasonal cycles.
    # Indexing safety: `K_seasons_to_average * season` is less than or equal to `n`,
    # so `context[-K_seasons_to_average * season:]` is always a valid slice.
    seasonal_data_for_averaging = context[-K_seasons_to_average * season:]
    
    # Reshape the data into (number_of_cycles, season_length) and compute the mean
    # across the cycles to get a robust hourly average for the day.
    # `np.nanmean` is used to handle potential NaNs within the historical data.
    # If a specific hour across K cycles is all NaN, its average will be NaN (which nan_to_num handles later).
    averaged_seasonal_pattern = np.nanmean(seasonal_data_for_averaging.reshape(K_seasons_to_average, season), axis=0)

    # Tile this averaged seasonal pattern to cover the entire prediction length.
    reps = int(np.ceil(prediction_length / season))
    forecasts = np.tile(averaged_seasonal_pattern, reps)[:prediction_length]

    # Hard requirement 4: Never output NaN or inf. Clip/guard as needed.
    # `np.nan_to_num` replaces any potential NaNs (e.g., if averaged_seasonal_pattern had NaNs
    # because all K values for a particular hour were NaN).
    # Use the robust `last_known_value` (calculated at the beginning) as the replacement.
    return np.nan_to_num(forecasts, nan=last_known_value).astype(float)