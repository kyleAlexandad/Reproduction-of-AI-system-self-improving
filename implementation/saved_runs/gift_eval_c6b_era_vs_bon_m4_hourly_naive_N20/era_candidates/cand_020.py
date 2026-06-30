import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # --- 1. Robust Fallback Value Determination ---
    # This value is crucial for handling empty, very short, or entirely NaN/Inf contexts,
    # and for filling missing values within the context or forecast.
    fallback_value = 0.0 # Default if no finite values are found.
    if n > 0:
        # Find indices of all finite values in the context.
        last_finite_idx = np.where(np.isfinite(context))[0]
        if len(last_finite_idx) > 0:
            # If finite values exist, the last one is the most reliable recent value.
            fallback_value = context[last_finite_idx[-1]]
    # Ensure fallback_value itself is finite, handling cases where context might contain only NaNs/Infs.
    if not np.isfinite(fallback_value):
        fallback_value = 0.0 # Fallback to 0.0 if the "last finite" was somehow non-finite itself.

    # Define the primary seasonal period for hourly data, which is daily (24 hours).
    season_daily = 24

    # --- 2. Handle very short contexts ---
    # If the context is empty, return an array of zeros to satisfy the output length requirement.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)
    
    # If the context is shorter than a full daily season, we cannot establish a reliable seasonal pattern.
    # In these cases, return a constant forecast using the robust fallback value.
    # This acts as a robust last-value naive forecast (or 0.0 if no valid last value).
    if n < season_daily:
        return np.full(prediction_length, fallback_value, dtype=float)

    # --- 3. Core Forecasting Logic: Averaged Seasonal Naive (Period 24) ---
    # This strategy improves upon simple seasonal naive by averaging the last K full daily cycles
    # to denoise the seasonal pattern, as recommended in the prompt.

    # Determine K: number of past full daily seasons to average.
    # The "GOOD SEASONAL TEMPLATE" suggests this calculation:
    # Use at least 1 full season, up to a maximum of 8 seasons, or as many as available in context.
    K = min(max(1, n // season_daily), 8)
    
    # Extract the data corresponding to the last K full seasons from the context.
    # This slice is guaranteed to be valid because `n >= season_daily` and `K >= 1`,
    # ensuring `K * season_daily` does not exceed `n`.
    data_for_averaging = context[-K * season_daily:]
    
    # Robustify the extracted data: replace any NaN or Inf values with the `fallback_value`.
    # This step is crucial for ensuring the seasonal pattern calculation is robust and produces finite values.
    cleaned_data_for_averaging = np.nan_to_num(
        data_for_averaging, nan=fallback_value, posinf=fallback_value, neginf=fallback_value
    )

    # Reshape the cleaned data into a K x season_daily matrix.
    # Each row in this matrix represents a full daily season.
    seasonal_matrix = cleaned_data_for_averaging.reshape(K, season_daily)
    
    # Calculate the mean across the rows (axis=0) to get the averaged seasonal pattern.
    # This results in a 1D array of length `season_daily`, where each element represents
    # the typical value for a specific hour of the day, averaged over `K` past days.
    averaged_seasonal_pattern = seasonal_matrix.mean(axis=0)

    # Tile the averaged seasonal pattern to cover the entire prediction length.
    # Calculate how many repetitions of the daily pattern are needed.
    num_repetitions = int(np.ceil(prediction_length / season_daily))
    
    # Create the full forecast by tiling the averaged pattern.
    # Then, truncate the tiled array to the exact `prediction_length`.
    forecast_output = np.tile(averaged_seasonal_pattern, num_repetitions)[:prediction_length]

    # --- 4. Final Robustness Check ---
    # As a final safeguard, ensure that no NaN or Inf values have inadvertently
    # appeared in the `forecast_output`. Replace them with `fallback_value` if any.
    # This step might be redundant if `cleaned_data_for_averaging` and `averaged_seasonal_pattern`
    # were perfectly clean, but it adds an extra layer of safety.
    final_forecast = np.nan_to_num(
        forecast_output, nan=fallback_value, posinf=fallback_value, neginf=fallback_value
    )
    
    # Return the final forecast as a float array of the specified prediction_length.
    return final_forecast.astype(float)