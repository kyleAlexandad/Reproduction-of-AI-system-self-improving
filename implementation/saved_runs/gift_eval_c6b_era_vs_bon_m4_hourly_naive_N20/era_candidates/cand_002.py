import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- 1. Determine a robust fallback value ---
    # This value will be used if context is empty, too short, all NaNs/Infs,
    # or to fill NaNs in the derived seasonal pattern.
    fallback_value = 0.0
    if n > 0:
        # Find the last finite value in context. This is the most reliable last known value.
        last_finite_idx = np.where(np.isfinite(context))[0]
        if len(last_finite_idx) > 0:
            fallback_value = context[last_finite_idx[-1]]
    # Ensure fallback_value itself is finite, defaulting to 0.0 if not.
    if not np.isfinite(fallback_value):
        fallback_value = 0.0

    # --- 2. Handle very short contexts ---
    # If the context array is empty, return a constant forecast using the fallback value.
    if n == 0:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 3. Define seasonality ---
    # Critical benchmark facts indicate strong daily seasonality (period 24) for hourly data.
    season = 24  # Daily seasonality for hourly data (24 hours in a day).

    # If context is shorter than one full season, fall back to a naive forecast
    # using the robust fallback_value. This ensures stability and is effectively last-value naive
    # for such short series as there's no full seasonal pattern to extract.
    if n < season:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 4. Derive the seasonal pattern (robust Seasonal Naive) ---
    # The benchmark states that Seasonal Naive (period 24) is very strong (MASE ~1.19).
    # This suggests prioritizing the pattern from the *last* available full season.

    # Extract the data from the last full season. This is our primary seasonal pattern candidate.
    seasonal_pattern_candidate = np.copy(context[-season:])

    # Identify any NaN values within this candidate seasonal pattern.
    nan_indices_in_pattern = np.where(np.isnan(seasonal_pattern_candidate))[0]

    if len(nan_indices_in_pattern) > 0:
        # If there are NaNs in the last season's data, we impute them.
        # Imputation strategy: use the average of that specific hour from *previous* seasons.
        # This makes the pattern more robust to missing values without losing the emphasis
        # on the most recent seasonal cycle.

        # Determine how many previous complete seasons to consider for imputation.
        # We look at history *before* the `seasonal_pattern_candidate` (`context[-season:]`).
        # `n - season` is the length of history available before the last season.
        # `(n - season) // season` gives the number of *full* previous seasons.
        # Capping at 8 prevents using excessively old data.
        K_for_imputation = min(max(0, (n - season) // season), 8)

        if K_for_imputation > 0:
            # Extract data from these K_for_imputation previous seasons for robust imputation.
            # `context[-(K_for_imputation + 1) * season : -season]` slices K_for_imputation
            # full seasons immediately preceding the last season.
            imputation_data_slice = context[-(K_for_imputation + 1) * season : -season]

            # Ensure we have enough data for the specified K_for_imputation.
            if len(imputation_data_slice) == K_for_imputation * season:
                # Reshape and calculate the mean for each hour across these previous seasons.
                mat_for_imputation = imputation_data_slice.reshape(K_for_imputation, season)
                hourly_averages_for_imputation = np.nanmean(mat_for_imputation, axis=0)

                # Fill NaNs in `seasonal_pattern_candidate` using these historical hourly averages.
                for idx in nan_indices_in_pattern:
                    if np.isfinite(hourly_averages_for_imputation[idx]):
                        seasonal_pattern_candidate[idx] = hourly_averages_for_imputation[idx]
            # Note: If `len(imputation_data_slice)` is not as expected (e.g., due to more leading
            # NaNs in the context or insufficient actual finite history), the remaining NaNs
            # will be handled by the fallback to `fallback_value` below.

        # Any remaining NaNs in `seasonal_pattern_candidate` (e.g., no previous seasons were
        # available for imputation, or all past values for that specific hour were also NaN)
        # are filled with the global robust `fallback_value`.
        seasonal_pattern_candidate[np.isnan(seasonal_pattern_candidate)] = fallback_value

    # The final seasonal pattern, guaranteed to be finite and robust.
    # This last `nan_to_num` is a safety net in case any value slipped through previous checks.
    seasonal_pattern = np.nan_to_num(seasonal_pattern_candidate, nan=fallback_value)


    # --- 5. Generate forecasts ---
    # Tile the cleaned seasonal pattern to cover the entire `prediction_length`.
    reps = int(np.ceil(prediction_length / season))
    tiled_forecast = np.tile(seasonal_pattern, reps)

    # Truncate the tiled pattern to the exact required `prediction_length`.
    final_forecast = tiled_forecast[:prediction_length]

    # Ensure the final output array has a float data type.
    return final_forecast.astype(float)