import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle SHORT context arrays robustly
    if n == 0:
        # Hard requirement 4: Never output NaN or inf.
        return np.zeros(prediction_length, dtype=float)

    # Critical Benchmark Fact: Hourly data has strong daily seasonality (period 24).
    season = 24

    # Fallback for contexts shorter than one full season
    if n < season:
        # Hard requirement 4: Never output NaN or inf.
        # Hard requirement 8: Sane fallback (last value naive)
        return np.full(prediction_length, float(context[-1]))

    # Determine K, the number of past full seasons to average.
    # Capped at 8 to prevent using too distant history, which might not be relevant.
    # Ensures K is at least 1 since n >= season at this point.
    K = min(max(1, n // season), 8)

    # Extract the last K full seasons from the context.
    # Hard requirement 9: INDEXING SAFETY - slicing from the end.
    # This is safe because n >= K * season due to the K calculation (n // season >= K).
    # Reshape into a matrix where each row is a full season.
    mat = context[-K * season:].reshape(K, season)

    # Calculate the average seasonal pattern across these K past seasons.
    # This provides a denoised seasonal profile for each "hour of the day".
    seasonal_pattern = mat.mean(axis=0)

    # --- Improvement: Add a level adjustment ---
    # The seasonal pattern averages multiple seasons. However, the overall level of the time
    # series might have drifted recently. We want to align our forecast to the current level.

    # 1. Calculate the mean level of the most recent full season available in the context.
    # This serves as an indicator of the current overall level of the series.
    # This is safe because n >= season is already checked.
    recent_level = context[-season:].mean()

    # 2. Calculate the average level of the derived 'seasonal_pattern'.
    # This is the baseline average level inherent in the seasonal profile.
    base_seasonal_level = seasonal_pattern.mean()

    # 3. Compute the level adjustment.
    # This difference tells us how much the current level (recent_level) deviates from
    # the averaged seasonal pattern's level (base_seasonal_level).
    # If K=1, then seasonal_pattern is just context[-season:], so recent_level == base_seasonal_level
    # and level_adjustment will be 0, effectively reducing to seasonal naive on the last season.
    # This is a desired robust behavior for short histories.
    level_adjustment = recent_level - base_seasonal_level

    # 4. Apply the level adjustment to the seasonal pattern.
    # This shifts the entire seasonal pattern up or down to match the recent overall level.
    adjusted_seasonal_pattern = seasonal_pattern + level_adjustment

    # --- Generate Forecast ---
    # Tile the adjusted seasonal pattern to cover the entire prediction_length.
    reps = int(np.ceil(prediction_length / season))
    forecast_output = np.tile(adjusted_seasonal_pattern, reps)[:prediction_length]

    # Hard requirement 4: Never output NaN or inf.
    # Use context[-1] as a fallback for any potential NaN values (though unlikely with means).
    # Hard requirement 3: Output length must be EXACTLY prediction_length.
    return np.nan_to_num(forecast_output, nan=float(context[-1])).astype(float)