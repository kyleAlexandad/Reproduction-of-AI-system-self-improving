import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: Hourly data has strong daily seasonality (period 24).
    season = 24

    # --- Fallbacks for very short contexts ---
    if n < season:
        # Fallback 1: Last-value naive if context is shorter than a full daily season.
        # This handles cases where seasonality cannot be extracted.
        # Hard requirement 9: Indexing safety. context[-1] is safe because n >= 1.
        return np.full(prediction_length, float(context[-1]), dtype=float)

    # --- Core Seasonal Forecast with K-averaging ---
    # The prompt recommends "average the last K full days at each hour to denoise the pattern".
    # This approach is generally more robust than just the last full season (seasonal naive)
    # when there's some noise, and is suggested by the provided template.

    # K: Number of past full seasons to average.
    # We take a minimum of 1 (always average at least one full season if available)
    # and a maximum of 8 (to limit the influence of very old data, as recent patterns are often more relevant).
    # Also ensures K * season does not exceed n.
    K = min(max(1, n // season), 8)

    # Extract the relevant historical data: the last K full seasons.
    # Hard requirement 9: Indexing safety. K * season <= n is guaranteed by the definition of K.
    # For example, if n = 25 and season = 24, then n // season = 1, so K = 1. We take context[-24:].
    # If n = 50 and season = 24, then n // season = 2, so K = 2. We take context[-48:].
    relevant_history = context[-K * season:]

    # Reshape the relevant history into K rows (each row is one full season) and 'season' columns.
    # This allows easy averaging across the 'K' seasons for each hour of the day.
    # The length of relevant_history is K * season, which is perfectly divisible by 'season'.
    seasonal_matrix = relevant_history.reshape(K, season)

    # Calculate the average pattern for each point in the season (e.g., average of all 1 AM values).
    # This gives us a single 'seasonal' pattern of length 'season'.
    seasonal_pattern = seasonal_matrix.mean(axis=0)

    # Tile this averaged seasonal pattern to cover the entire prediction_length.
    num_repetitions = int(np.ceil(prediction_length / season))
    forecast_values = np.tile(seasonal_pattern, num_repetitions)[:prediction_length]

    # Hard requirement 4: Never output NaN or inf.
    # np.nan_to_num replaces any NaNs (e.g., if context contained NaNs that propagated through mean calculations)
    # with a safe fallback value (the last observed value from context).
    # Finally, ensure the output type is float.
    return np.nan_to_num(forecast_values, nan=float(context[-1])).astype(float)