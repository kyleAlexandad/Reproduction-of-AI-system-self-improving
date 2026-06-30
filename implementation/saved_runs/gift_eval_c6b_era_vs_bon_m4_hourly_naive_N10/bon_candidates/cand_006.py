import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    if n == 0:
        # Hard requirement 4: Never output NaN or inf
        return np.zeros(prediction_length, dtype=float)

    # Daily seasonality for hourly data (period 24) is a critical hint for m4_hourly/H.
    season = 24

    # If context is shorter than one full season, fall back to last-value naive.
    # This aligns with the recommendation and hard requirement 8.
    if n < season:
        # Hard requirement 4: Never output NaN or inf
        # Hard requirement 3: Output length must be EXACTLY prediction_length
        return np.full(prediction_length, float(context[-1]))

    # --- Core Seasonal Averaging Logic ---
    # K: Number of full seasons to average for the seasonal pattern.
    # Limit K to max 8 seasons to balance robustness (more averaging) and responsiveness (less old data),
    # and ensure K is at least 1 if n >= season.
    K = min(max(1, n // season), 8)

    # Extract the last K full seasons from the context.
    # Hard requirement 9: INDEXING SAFETY - slice from the end to prevent out-of-bounds access.
    seasonal_data = context[-K * season:]
    
    # Reshape into a K x season matrix.
    # Each row now represents one full historical season.
    mat = seasonal_data.reshape(K, season)

    # Calculate the mean of each season (row-wise mean).
    # This represents the average level of each of the K historical seasons.
    mat_means = mat.mean(axis=1, keepdims=True)

    # Deseasonalize by subtracting the row-wise mean from each season.
    # This isolates the seasonal *deviations* from the mean level for each historical season.
    deseasonalized_mat = mat - mat_means

    # Average these deseasonalized patterns across K seasons to get a robust average seasonal shape.
    # This gives the average deviation for each point within a season, centered around zero.
    seasonal_deviations = deseasonalized_mat.mean(axis=0)

    # Determine the recent base level for the forecast.
    # Using the mean of the very last full season (context[-season:]) to anchor the forecast
    # level to the most recent history. This makes the forecast more responsive to recent
    # shifts in the overall level compared to using an average over K seasons for the level.
    recent_base_level = context[-season:].mean()
    
    # Repeat the seasonal deviations pattern to cover the entire prediction length.
    reps = int(np.ceil(prediction_length / season))
    tiled_seasonal_deviations = np.tile(seasonal_deviations, reps)[:prediction_length]

    # Combine the recent base level with the tiled seasonal deviations to get the final forecast.
    # This projects the average seasonal shape onto the most recent observed level.
    out = recent_base_level + tiled_seasonal_deviations

    # Hard requirement 4: Never output NaN or inf.
    # Use nan_to_num as a robust guard. If any part of the calculation resulted in NaN (e.g.,
    # due to extreme input values or division by zero, though unlikely here), it will be
    # replaced by the last known context value. This ensures a clean numerical output.
    # Hard requirement 3: Output length must be EXACTLY prediction_length
    return np.nan_to_num(out, nan=float(context[-1])).astype(float)