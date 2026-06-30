import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle SHORT context arrays robustly
    # If no data is available, return an array of zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical Benchmark Fact: Hourly data (freq "H") has strong DAILY seasonality, period 24.
    season = 24

    # Recommended Strategy: Fall back to last-value naive when context is too short
    # to extract at least one full seasonal period.
    if n < season:
        # Hard requirement 3: Output length exactly prediction_length
        # Hard requirement 4: Never output NaN or inf. Use the last observed value.
        return np.full(prediction_length, float(context[-1]))

    # Recommended Strategy: Use a seasonal backbone.
    # Average the last K full days (seasonal periods) to denoise the seasonal pattern.
    # K: number of full seasons to average.
    # - `max(1, n // season)` ensures we use at least one full season if available,
    #   and uses all available full seasons if `n` is larger.
    # - `min(..., 8)` caps K at 8 to prevent using very old data which might be stale.
    K = min(max(1, n // season), 8)
    
    # Hard requirement 9: INDEXING SAFETY. Slice from the END.
    # Extract the most recent K full seasonal periods from the context.
    # Since `K <= n // season`, `K * season <= n` is guaranteed,
    # so `context[-K * season:]` is a safe slice.
    recent_seasonal_data = context[-K * season:]
    
    # Reshape the data into a matrix where each row represents a full season (e.g., a day),
    # and each column corresponds to a specific point within the season (e.g., an hour of the day).
    mat = recent_seasonal_data.reshape(K, season)
    
    # Calculate the average for each point within the season across the K periods.
    # This gives us the robust average seasonal pattern.
    seasonal_pattern = mat.mean(axis=0) # This will be a 1D array of length `season`.

    # Project this calculated seasonal pattern over the entire prediction horizon.
    # Determine how many repetitions of the seasonal pattern are needed.
    reps = int(np.ceil(prediction_length / season))
    
    # Tile the seasonal pattern to cover the required repetitions.
    tiled_forecast = np.tile(seasonal_pattern, reps)
    
    # Trim the tiled forecast to the exact `prediction_length`.
    # Hard requirement 3: Output length must be EXACTLY prediction_length.
    out = tiled_forecast[:prediction_length]

    # Hard requirement 4: Never output NaN or inf.
    # `np.nan_to_num` replaces any NaN values with the last observed context value
    # and infinities with large finite numbers, ensuring a robust output.
    # `.astype(float)` ensures the final array data type is float.
    return np.nan_to_num(out, nan=float(context[-1])).astype(float)
