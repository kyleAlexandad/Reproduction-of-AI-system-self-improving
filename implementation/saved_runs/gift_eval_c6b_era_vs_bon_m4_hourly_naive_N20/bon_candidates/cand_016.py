import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle SHORT context arrays robustly
    # If no context is available, return zeros for all predictions.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical Benchmark Fact: Hourly data has strong DAILY seasonality (period 24).
    season = 24

    # If the context is shorter than one full season,
    # fall back to the last-value naive forecast.
    if n < season:
        # Hard requirement 4: Never output NaN or inf. context[-1] is guaranteed to exist and be float.
        return np.full(prediction_length, float(context[-1]))

    # CRITICAL BENCHMARK FACT: Exploit seasonality; average the last K full days.
    # K determines how many past full seasons (days) to average.
    # We limit K to a maximum of 8 seasons to prevent using very old data,
    # and ensure at least 1 full season is used if available.
    # (n // season) gives the number of full seasons available in the context.
    K = min(max(1, n // season), 8)

    # Hard requirement 9: INDEXING SAFETY.
    # context[-K * season:] safely extracts the last K full seasons.
    # This is safe because K is capped at n // season, meaning K * season <= n.
    relevant_context = context[-K * season:]
    
    # Reshape the relevant context into a matrix where each row represents a full season.
    # Example: If K=2, season=24, mat will be a 2x24 array.
    mat = relevant_context.reshape(K, season)
    
    # Calculate the average seasonal pattern by taking the mean across the rows.
    # This gives a 1D array of length 'season', representing the typical values for each hour of the day.
    seasonal_pattern = mat.mean(axis=0)

    # RECOMMENDED STRATEGY: Add a small, damped level/trend correction.
    # Here, we implement a simple level adjustment to anchor the seasonal forecast
    # to the most recent observation. This helps to account for recent shifts in the series level.
    
    # Get the last observed value from the context.
    last_observed_value = context[-1]
    
    # Determine the index within the seasonal cycle for the last observed value.
    # (n - 1) gives the 0-based index of the last element in context.
    last_seasonal_index = (n - 1) % season
    
    # Get the corresponding value from our averaged seasonal pattern.
    last_seasonal_value_from_pattern = seasonal_pattern[last_seasonal_index]
    
    # Calculate the difference between the actual last observed value and its
    # seasonal pattern prediction. This difference is our level adjustment.
    level_adjustment = last_observed_value - last_seasonal_value_from_pattern
    
    # Apply this level adjustment to the entire seasonal pattern.
    # This effectively shifts the entire seasonal forecast up or down to align
    # with the most recent data point.
    seasonal_forecast_base = seasonal_pattern + level_adjustment

    # Generate the final forecast by tiling the adjusted seasonal pattern.
    # Calculate how many repetitions of the seasonal pattern are needed to cover the prediction_length.
    reps = int(np.ceil(prediction_length / season))
    
    # Tile the seasonal_forecast_base.
    tiled_forecast = np.tile(seasonal_forecast_base, reps)
    
    # Hard requirement 3: Output length must be EXACTLY prediction_length.
    # Clip the tiled forecast to the required length.
    out = tiled_forecast[:prediction_length]

    # Hard requirement 4: Never output NaN or inf.
    # Use nan_to_num to replace any potential NaNs (though unlikely with this method)
    # with the last observed value, and ensure the output is float.
    return np.nan_to_num(out, nan=last_observed_value).astype(float)