import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard requirement 8: Handle short context arrays
    # If no historical data, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Define seasonal periods relevant for hourly data.
    # Daily seasonality is typically 24 hours.
    # Weekly seasonality is 24 hours * 7 days = 168 hours.
    s24 = 24  
    s168 = 168

    # --- Fallback for very short context ---
    # If there isn't enough data for even one full daily season,
    # fall back to a simple last-value naive forecast.
    if n < s24:
        return np.full(prediction_length, float(context[-1]))

    # --- Calculate Daily Seasonal Pattern (P24) ---
    # Average the last K full daily cycles to get a robust daily pattern.
    # K_24 is capped at 8 to avoid using data that is too old and potentially irrelevant,
    # and ensured to be at least 1 since we've already checked n >= s24.
    K_24 = min(max(1, n // s24), 8)
    
    # Slice the context to get the last K_24 daily cycles.
    # This slice is guaranteed to be valid because n >= s24 and K_24 * s24 <= n.
    mat_24 = context[-K_24 * s24:].reshape(K_24, s24)
    # The mean across rows gives the average pattern for each hour of the day.
    seasonal_24 = mat_24.mean(axis=0)

    # Initialize the base seasonal pattern with the daily pattern.
    base_seasonal_pattern = seasonal_24

    # --- If enough data, incorporate Weekly Seasonal Pattern (P168) ---
    # A weekly pattern provides more specific forecasts (e.g., Monday 9 AM vs. Tuesday 9 AM).
    # This is only possible if at least one full weekly cycle is present in the context.
    if n >= s168:
        # Average the last K full weekly cycles. K_168 is capped at 4.
        K_168 = min(max(1, n // s168), 4)
        
        # Slice the context to get the last K_168 weekly cycles.
        # This slice is guaranteed to be valid.
        mat_168 = context[-K_168 * s168:].reshape(K_168, s168)
        # The mean across rows gives the average pattern for each hour of the week.
        seasonal_168 = mat_168.mean(axis=0)

        # Blend the daily and weekly patterns.
        # The `blend_factor` smoothly transitions from favoring the daily pattern (when `n` is just enough for a week)
        # to favoring the weekly pattern (as more weekly data becomes available).
        # `max_transition_cycles` defines how many weekly cycles beyond the first are needed for a full transition.
        max_transition_cycles = 2 
        blend_factor = min(1.0, (n - s168) / (s168 * max_transition_cycles))
        
        # Extend the daily seasonal pattern to the length of a week (168 hours)
        # by tiling it 7 times (168 / 24 = 7).
        seasonal_24_extended = np.tile(seasonal_24, s168 // s24)

        # Combine the two patterns using the blend factor.
        # When `blend_factor` is 0 (i.e., `n` is exactly `s168`), it uses `seasonal_24_extended`.
        # When `blend_factor` is 1 (i.e., `n` is `s168 * (1 + max_transition_cycles)` or more), it uses `seasonal_168`.
        base_seasonal_pattern = (blend_factor * seasonal_168 + 
                                 (1.0 - blend_factor) * seasonal_24_extended)
    
    # --- Generate Forecast ---
    # The last known data point is at index `n-1`.
    # The first forecast point corresponds to index `n` in the overall time series.
    # We need to map these absolute time series indices to positions within our `base_seasonal_pattern`.
    
    pattern_length = len(base_seasonal_pattern)
    
    # Calculate the indices within the seasonal pattern for each forecast step.
    # `np.arange(prediction_length)` gives 0, 1, ..., prediction_length-1 (representing relative forecast steps).
    # Adding `n` converts these to absolute indices in the full time series context.
    # The modulo `pattern_length` maps these absolute indices to the correct position within `base_seasonal_pattern`.
    forecast_indices_in_pattern = (np.arange(prediction_length) + n) % pattern_length
    
    # Apply the determined seasonal pattern to generate the raw forecasts.
    out = base_seasonal_pattern[forecast_indices_in_pattern]

    # Hard requirement 4: Never output NaN or inf. Clip/guard as needed.
    # In the unlikely event that `base_seasonal_pattern` contains NaNs (e.g., from an edge case
    # where all values in a seasonal average were NaN, though our setup minimizes this risk),
    # replace them with a sensible fallback like the last context value.
    out = np.nan_to_num(out, nan=float(context[-1])).astype(float)
    
    return out