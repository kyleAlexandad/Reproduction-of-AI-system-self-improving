import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: Hourly data has strong daily seasonality (period 24).
    season_daily = 24

    # --- Core Daily Seasonal Forecast (K-averaged) ---
    if n < season_daily:
        # Fallback to last-value naive if context is shorter than a full daily season.
        # This is also a safe fallback for cases where season_daily is not applicable.
        # Hard requirement 4 & 9: Safe fallback.
        base_forecast = np.full(prediction_length, float(context[-1]), dtype=float)
    else:
        # Average the last K full daily seasons to get a more robust seasonal pattern.
        # The cap of 8 is a common heuristic to avoid averaging too much historical data
        # which might be stale. `n // season_daily` ensures we only use full seasons.
        K_daily = min(max(1, n // season_daily), 8)
        
        # Hard requirement 9: Indexing safety.
        # Slice exactly K_daily * season_daily points from the end of the context.
        start_idx_daily = n - K_daily * season_daily
        
        # Reshape into a (K_daily, season_daily) matrix, then average across rows (days).
        daily_seasonal_matrix = context[start_idx_daily:].reshape(K_daily, season_daily)
        averaged_daily_pattern = daily_seasonal_matrix.mean(axis=0)
        
        # Tile the pattern to cover the prediction_length.
        num_repetitions = int(np.ceil(prediction_length / season_daily))
        base_forecast = np.tile(averaged_daily_pattern, num_repetitions)[:prediction_length]

        # --- Damped Level Correction ---
        # The prompt suggests "optionally add a small, damped level/trend correction".
        # Instead of a trend (which the benchmark warns against), we apply a level
        # adjustment based on the difference between the most recent season's average
        # and the K-averaged pattern's average. This corrects for recent shifts in
        # the overall level of the series.
        
        # Calculate the average level of the most recent full daily season.
        current_recent_avg = np.mean(context[-season_daily:])
        
        # Calculate the average level of the K-averaged seasonal pattern.
        pattern_avg = np.mean(averaged_daily_pattern)
        
        # The difference is a level offset.
        level_offset = current_recent_avg - pattern_avg
        
        # Apply this offset to the forecast, but damp it over time into the future.
        # A damping factor of 0.9 ensures the correction fades, reducing over-correction risk.
        damping_factor = 0.90 
        
        # Apply the damped level correction. The damping is applied per full season block
        # into the future (i.e., for each subsequent 24-hour period in the forecast horizon).
        for i in range(prediction_length):
            # i // season_daily gives the number of full seasons passed in the forecast horizon
            # e.g., 0 for i=0..23, 1 for i=24..47 etc.
            damped_level_contribution = level_offset * (damping_factor ** (i // season_daily))
            base_forecast[i] += damped_level_contribution

    # Hard requirement 4: Never output NaN or inf.
    # np.nan_to_num replaces NaNs with 0 by default, or with a specified value.
    # Here, context[-1] (the last observed value) is a safe fallback as n > 0 is guaranteed here.
    # Finally, ensure the output type is float.
    return np.nan_to_num(base_forecast, nan=float(context[-1])).astype(float)
