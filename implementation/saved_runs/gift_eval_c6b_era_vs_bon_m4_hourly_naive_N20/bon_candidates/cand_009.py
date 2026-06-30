import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- Robust NaN and inf fallback value determination ---
    # This value will be used if any forecast point results in NaN/inf.
    # We prioritize the last non-NaN/inf value from context, otherwise default to 0.0.
    fallback_val = 0.0
    if n > 0:
        if not np.isnan(context[-1]) and not np.isinf(context[-1]):
            fallback_val = float(context[-1])
        else:
            # If the last value is NaN/inf, try to find the last non-NaN/inf value in context
            valid_indices = np.where(~np.isnan(context) & ~np.isinf(context))[0]
            if len(valid_indices) > 0:
                fallback_val = float(context[valid_indices[-1]])
            # If all context values are NaN/inf, fallback_val remains 0.0

    # Hard requirement 8: Handle SHORT context arrays robustly
    if n == 0:
        # If context is empty, return an array filled with the fallback value.
        return np.full(prediction_length, fallback_val, dtype=float)

    season = 24  # Critical benchmark fact: Daily seasonality for hourly data (period 24)

    if n < season:
        # Fallback to last-value naive (or fallback_val if last_value is NaN/inf)
        # when not enough data for a full season.
        return np.full(prediction_length, fallback_val, dtype=float)

    # Use a seasonal backbone: average the last K full seasons (days)
    # K: number of full seasons to average. Max 8 seasons, min 1.
    # This ensures we use available data without going too far back and
    # also respects Hard requirement 9 (INDEXING SAFETY).
    K = min(max(1, n // season), 8)
    
    # Extract the last K full seasons from the context.
    # Using negative slicing `context[-X:]` is safe and robust.
    seasonal_data = context[-K * season:]
    
    # Reshape into K rows (number of seasons), `season` columns (hours in a day).
    # np.nanmean will ignore NaNs by default when calculating means.
    mat = seasonal_data.reshape(K, season)
    
    # Calculate the average seasonal pattern (mean for each hour of the day).
    # If a column (specific hour across K seasons) is all NaNs, its mean will be NaN.
    seasonal_pattern = np.nanmean(mat, axis=0)

    # Tile the seasonal pattern to cover the prediction length.
    reps = int(np.ceil(prediction_length / season))
    base_forecast = np.tile(seasonal_pattern, reps)[:prediction_length]

    # --- Add a small, damped level correction on top of the seasonal pattern ---
    # Determine the current seasonal index for the last observed value in context.
    current_seasonal_idx = (n - 1) % season
    
    # Get the actual last observed value and its expected seasonal counterpart.
    last_actual_value = context[-1]
    expected_seasonal_value = seasonal_pattern[current_seasonal_idx]

    level_correction = 0.0
    # Calculate the level correction only if both the last actual value and
    # its corresponding seasonal expectation are valid numbers.
    if not np.isnan(last_actual_value) and not np.isinf(last_actual_value) and \
       not np.isnan(expected_seasonal_value) and not np.isinf(expected_seasonal_value):
        # This represents a recent level shift: how much the last point deviates from its expected seasonal value.
        level_correction = last_actual_value - expected_seasonal_value
    # If either is NaN/inf, level_correction remains 0.0, effectively disabling this correction.
    
    # Apply a damped correction: linearly decay the correction to zero over the horizon.
    # The damping factor decreases from 1 (at the first forecast step) to ~0 (at the last step).
    damping_factor_array = (prediction_length - np.arange(prediction_length)) / prediction_length
    
    # Apply the damped level correction to the base seasonal forecast.
    forecast_with_correction = base_forecast + level_correction * damping_factor_array

    # Hard requirement 4: Never output NaN or inf. Clip/guard as needed.
    # np.nan_to_num replaces NaNs/infs with the determined `fallback_val`.
    out = np.nan_to_num(forecast_with_correction, nan=fallback_val, posinf=fallback_val, neginf=fallback_val).astype(float)
    
    # Hard requirement 3: Output length must be EXACTLY prediction_length.
    return out