import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays robustly
    # If no data, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark fact: daily seasonality (period 24) for hourly data.
    # Weekly seasonality (period 168) may also help.
    season_daily = 24
    season_weekly = 168

    # Fallback to last-value naive if there isn't enough data for one full primary season.
    if n < season_daily:
        return np.full(prediction_length, float(context[-1]))

    # --- 1. Calculate a robust daily seasonal pattern (K-averaged) ---
    # Average the last K full daily cycles to denoise the pattern.
    # K is capped at 8 to prevent using too much history for potentially non-stationary series.
    K_days = min(max(1, n // season_daily), 8)
    
    # Ensure there is enough data for at least one full daily cycle (guaranteed by 'n < season_daily' check).
    # Slice the last K_days * season_daily points, reshape to (K_days, season_daily), and average.
    daily_seasonal_pattern = context[-K_days * season_daily:].reshape(K_days, season_daily).mean(axis=0)

    # Determine the starting index for the daily pattern for the first forecast point (h=0).
    # If n is the current length, context[n-1] is the last observed point.
    # The first forecast point corresponds to index n in the overall series.
    start_idx_daily = n % season_daily

    # Initialize forecasts with the daily seasonal pattern.
    forecasts = np.zeros(prediction_length, dtype=float)
    for h in range(prediction_length):
        forecasts[h] = daily_seasonal_pattern[(start_idx_daily + h) % season_daily]

    # --- 2. Incorporate weekly seasonal pattern (if enough data) ---
    # This is an optional enhancement, blended with the daily pattern.
    # Average the last K full weekly cycles. K is capped at 4 as weekly data is longer.
    K_weeks = min(max(1, n // season_weekly), 4)

    # Only attempt to calculate and blend if there's sufficient data for at least one full K_weeks cycle.
    if n >= K_weeks * season_weekly:
        weekly_seasonal_pattern = context[-K_weeks * season_weekly:].reshape(K_weeks, season_weekly).mean(axis=0)
        
        # Determine the starting index for the weekly pattern.
        start_idx_weekly = n % season_weekly

        # Blend daily and weekly patterns using a weighted average.
        # Daily seasonality is typically stronger for hourly data, so it gets a higher weight.
        weight_daily = 0.8
        weight_weekly = 0.2
        
        for h in range(prediction_length):
            daily_comp = daily_seasonal_pattern[(start_idx_daily + h) % season_daily]
            weekly_comp = weekly_seasonal_pattern[(start_idx_weekly + h) % season_weekly]
            forecasts[h] = weight_daily * daily_comp + weight_weekly * weekly_comp

    # --- 3. Add a damped level/trend correction ---
    # This requires at least two full primary seasonal cycles (2 * season_daily) to estimate
    # a trend from the last value compared to its seasonal counterpart in the previous cycle.
    if n >= 2 * season_daily:
        # Estimate the recent change in level over one daily seasonal period.
        # This captures if the series is trending up or down overall, relative to its seasonal pattern.
        # Indexing safety: context[-1] is safe if n >= 1. context[-1 - season_daily] is safe if n >= season_daily + 1.
        # The condition n >= 2 * season_daily ensures both indices are valid.
        level_trend_over_season = context[-1] - context[-1 - season_daily]

        # Apply exponential damping to the trend. This prevents over-extrapolation.
        damping_factor = 0.95 
        
        # Apply the damped level trend to each forecast step.
        # (h + 1) is used because h=0 corresponds to the first step into the future.
        for h in range(prediction_length):
            forecasts[h] += level_trend_over_season * (damping_factor ** (h + 1))

    # Hard requirement 4: Never output NaN or inf.
    # Replace any potential NaNs or Infs with a fallback value.
    # The last observed value is a reasonable fallback; otherwise, 0.0 for truly empty context.
    fallback_value = float(context[-1]) if n > 0 else 0.0
    return np.nan_to_num(forecasts, nan=fallback_value, posinf=fallback_value, neginf=fallback_value).astype(float)