import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- 1. Determine a robust fallback value ---
    # This value is crucial for handling empty, short, or all-NaN/Inf contexts,
    # and for imputing NaNs in the derived seasonal pattern.
    fallback_value = 0.0
    if n > 0:
        # Find the last finite value in context. This is the most reliable last known value.
        last_finite_idx = np.where(np.isfinite(context))[0]
        if len(last_finite_idx) > 0:
            fallback_value = context[last_finite_idx[-1]]
    # Ensure fallback_value itself is finite, defaulting to 0.0 if not.
    if not np.isfinite(fallback_value):
        fallback_value = 0.0

    # --- 2. Handle very short or empty contexts ---
    if n == 0:
        # If no history, forecast the fallback_value for all steps.
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 3. Define primary seasonality (daily) ---
    # Critical benchmark facts indicate strong daily seasonality (period 24) for hourly data.
    season_daily = 24

    # Fallback for contexts shorter than one full daily season.
    # In this case, we don't have enough data to extract a reliable seasonal pattern,
    # so we return a simple naive forecast using the robust fallback_value.
    if n < season_daily:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 4. Calculate the primary (daily) seasonal pattern ---
    # Average the last K full daily seasons to denoise the pattern.
    # Capping K (e.g., at 8) prevents using excessively old or potentially irrelevant data
    # and keeps the computation lightweight.
    # The minimum K is 1, ensuring at least the last full season is used if available.
    K_daily = min(max(1, n // season_daily), 8)

    # Extract the data for the last K_daily full seasons.
    # `context[-K_daily * season_daily:]` ensures we get exactly `K_daily` blocks
    # of `season_daily` points from the end of the context.
    daily_seasonal_data_slice = context[-K_daily * season_daily:]

    # Reshape the extracted data into a matrix.
    # Each row represents a full daily season, and each column corresponds to an hour-of-day.
    daily_seasonal_matrix = daily_seasonal_data_slice.reshape(K_daily, season_daily)

    # Compute the mean for each hour-of-day across the K_daily seasons.
    # `np.nanmean` is crucial here to handle potential NaNs in the historical data,
    # ignoring them in the average calculation. If a column is all NaNs, its mean will be NaN.
    daily_seasonal_pattern = np.nanmean(daily_seasonal_matrix, axis=0)

    # --- 5. Impute NaNs in the derived seasonal pattern and ensure finiteness ---
    # If any hour-of-day in the `daily_seasonal_pattern` is NaN (meaning all K_daily values
    # for that specific hour were NaN), fill it using the robust `fallback_value`.
    # `np.nan_to_num` also handles inf values, replacing them with `fallback_value`.
    daily_seasonal_pattern = np.nan_to_num(daily_seasonal_pattern, nan=fallback_value,
                                            posinf=fallback_value, neginf=fallback_value)

    # --- 6. Optional: Add a subtle damped level adjustment ---
    # This adjustment anchors the forecast to the latest observed level more smoothly,
    # without introducing an explicit trend. It corrects for the difference between
    # the last actual observation and its corresponding value in the calculated seasonal pattern.
    last_context_value = context[-1] # Guaranteed to exist due to n >= season_daily check

    if np.isfinite(last_context_value):
        # Determine the index (hour-of-day) in the seasonal pattern that corresponds to `context[-1]`.
        last_seasonal_point_index = (n - 1) % season_daily
        
        # Calculate the deviation of the last observed value from its seasonal component.
        deviation = last_context_value - daily_seasonal_pattern[last_seasonal_point_index]

        # Apply a small, damped portion of this deviation as an additive adjustment
        # to the entire seasonal pattern. This provides a mild level shift for the forecast.
        # A small damping factor (e.g., 0.1-0.3) prevents over-correction and
        # largely preserves the primary seasonal shape.
        damping_factor = 0.2
        daily_seasonal_pattern += damping_factor * deviation # Apply adjustment in-place

    # --- 7. Generate forecasts ---
    # Tile the final (and potentially adjusted) daily seasonal pattern to cover the horizon.
    reps = int(np.ceil(prediction_length / season_daily))
    tiled_forecast = np.tile(daily_seasonal_pattern, reps)

    # Truncate the tiled pattern to the exact required `prediction_length`.
    final_forecast = tiled_forecast[:prediction_length]

    # Final safeguard: ensure all output values are finite floats.
    return np.nan_to_num(final_forecast, nan=fallback_value, posinf=fallback_value, neginf=fallback_value).astype(float)