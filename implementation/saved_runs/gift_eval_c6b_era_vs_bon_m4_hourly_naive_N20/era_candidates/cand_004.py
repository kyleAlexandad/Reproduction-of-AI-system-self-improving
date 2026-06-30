import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- 1. Determine a robust fallback value ---
    # This value is crucial for handling empty/short contexts, or NaNs in patterns.
    # It will be the last *finite* value in the context, or 0.0 if no finite values exist.
    fallback_value = 0.0
    if n > 0:
        last_finite_idx = np.where(np.isfinite(context))[0]
        if len(last_finite_idx) > 0:
            fallback_value = context[last_finite_idx[-1]]
        # If all values in context are non-finite, fallback_value remains 0.0, which is fine.
    # Final safety check to ensure fallback_value is finite.
    if not np.isfinite(fallback_value):
        fallback_value = 0.0

    # --- 2. Handle very short contexts ---
    # If the context array is empty, return a constant forecast using the fallback value.
    if n == 0:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 3. Define seasonal periods ---
    daily_season = 24    # Daily seasonality for hourly data (24 hours in a day)
    weekly_season = 168  # Weekly seasonality for hourly data (7 days * 24 hours)

    # --- 4. Select the dominant seasonality and `K` for averaging based on available context ---
    current_season_period = daily_season # Default to daily seasonality
    K = 1 # Default K, will be updated based on available full periods

    if n >= weekly_season:
        current_season_period = weekly_season
        # For weekly seasonality, average up to the last 4 full weeks.
        # This keeps the pattern relatively recent and computationally light,
        # while providing more robustness than just one week.
        K = min(max(1, n // weekly_season), 4)
    elif n >= daily_season:
        current_season_period = daily_season
        # For daily seasonality, average up to the last 8 full days.
        # This provides good denoising for strong daily patterns.
        K = min(max(1, n // daily_season), 8)
    else:
        # Context is too short even for a single daily season.
        # Fall back to a simple naive forecast using the robust fallback_value.
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 5. Derive the seasonal pattern using K-period averaging ---
    # The `context` slice `[-K * current_season_period:]` extracts the last K full periods.
    # This slice is guaranteed to be valid because `K * current_season_period <= n`
    # by how `K` is calculated (`K <= n // current_season_period`).
    history_for_season_pattern = context[-K * current_season_period:]

    # Reshape the history into K rows (each representing a full period) and
    # `current_season_period` columns (each representing a point within the period).
    # `np.nanmean` computes the mean for each column, ignoring NaN values.
    seasonal_pattern_candidate = np.nanmean(
        history_for_season_pattern.reshape(K, current_season_period),
        axis=0
    )

    # --- 6. Robustly handle any remaining NaNs in the derived seasonal pattern ---
    # If, for a specific position within the season, all K corresponding historical values were NaN,
    # then `nanmean` would produce NaN for that position.
    # These remaining NaNs are filled with the robust fallback_value to ensure finite output.
    seasonal_pattern = np.nan_to_num(seasonal_pattern_candidate, nan=fallback_value)

    # --- 7. Generate forecasts ---
    # Tile the robust seasonal pattern to cover the entire `prediction_length`.
    # `np.ceil` ensures we generate enough repetitions.
    reps = int(np.ceil(prediction_length / current_season_period))
    tiled_forecast = np.tile(seasonal_pattern, reps)

    # Truncate the tiled pattern to the exact required `prediction_length`.
    final_forecast = tiled_forecast[:prediction_length]

    # Ensure the final output array has a float data type, as required.
    return final_forecast.astype(float)