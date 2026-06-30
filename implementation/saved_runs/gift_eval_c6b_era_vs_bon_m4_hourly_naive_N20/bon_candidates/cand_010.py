import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Determine a safe fallback value for NaNs or empty context.
    # If the last value is NaN, try the mean of non-NaN values. If all NaNs, use 0.0.
    fallback_fill_value = 0.0
    if n > 0:
        if np.isnan(context[-1]):
            non_nan_context = context[~np.isnan(context)]
            if len(non_nan_context) > 0:
                fallback_fill_value = non_nan_context.mean()
            # Else, fallback_fill_value remains 0.0
        else:
            fallback_fill_value = context[-1]
    
    # Handle empty context by returning all zeros or fallback value
    if n == 0:
        return np.full(prediction_length, fallback_fill_value, dtype=float)

    # For hourly data, daily seasonality (24 hours) is critical.
    season_daily = 24

    # Fallback to last-value naive if context is too short for a full seasonal cycle
    if n < season_daily:
        return np.full(prediction_length, fallback_fill_value, dtype=float)

    # Calculate K, the number of past full seasons to average.
    # Limit K to a reasonable maximum (e.g., 8 days) to avoid using too old data,
    # and ensure at least one full season (K=1).
    K_daily = min(max(1, n // season_daily), 8)

    # Extract the relevant context segment for seasonal averaging.
    # This ensures we get K_daily full seasons from the end of the context.
    seasonal_data_segment = context[-K_daily * season_daily:]

    # Reshape the segment to (K, season) and compute the mean across seasons (axis=0).
    # This yields the averaged seasonal pattern.
    mat_daily = seasonal_data_segment.reshape(K_daily, season_daily)
    seasonal_daily_pattern = mat_daily.mean(axis=0)

    # Calculate a level adjustment.
    # This re-centers the averaged seasonal pattern based on the most recent full season's average.
    # This helps in adapting to recent level shifts in the series.
    recent_season_avg = np.mean(context[-season_daily:])
    seasonal_pattern_avg = np.mean(seasonal_daily_pattern)
    level_shift = recent_season_avg - seasonal_pattern_avg

    # Generate base forecasts by tiling the seasonal pattern.
    # Calculate how many repetitions are needed to cover the prediction_length.
    reps = int(np.ceil(prediction_length / season_daily))
    base_forecast = np.tile(seasonal_daily_pattern, reps)[:prediction_length]

    # Apply the calculated level shift to the base forecast.
    final_forecast = base_forecast + level_shift

    # Ensure no NaNs or Infs in the output, replacing NaNs with the robust fallback value.
    # np.nan_to_num also handles +/- inf by replacing them with very large/small floats.
    final_forecast = np.nan_to_num(final_forecast, nan=fallback_fill_value)

    return final_forecast.astype(float)