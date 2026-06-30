import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Determine a robust fallback value to use in case of empty context,
    # short context, or propagation of NaNs/Infs into the forecast.
    # This value is guaranteed to be a finite float.
    fallback_value = 0.0  # Default if no valid data is found
    if n > 0:
        # Find finite (non-NaN, non-inf) values in the context.
        finite_context = context[np.isfinite(context)]
        if len(finite_context) > 0:
            # If there are finite values, use their mean as a robust fallback.
            fallback_value = float(np.mean(finite_context))
        # If all context values are non-finite (NaN or Inf), fallback_value remains 0.0.

    # Hard requirement 8: Handle short context arrays robustly.
    # If no history, return the determined fallback value.
    if n == 0:
        return np.full(prediction_length, fallback_value, dtype=float)

    # Critical benchmark fact: Hourly data has strong daily seasonality (period 24).
    season_daily = 24  # Daily seasonality for hourly data

    # Fallback 1: If context is shorter than a full daily season,
    # seasonal patterns cannot be reliably extracted. Use the fallback value.
    if n < season_daily:
        return np.full(prediction_length, fallback_value, dtype=float)

    # Core forecasting strategy: Average the last K full daily seasons.
    # This denoises the seasonal pattern compared to just using the very last season.
    # The prompt's "GOOD SEASONAL TEMPLATE" suggests this approach.
    
    # K: Number of past full seasons to average.
    # We ensure K is at least 1 (to always have a pattern) and at most 8 (to limit memory/computation
    # and avoid using very old, potentially irrelevant patterns).
    K = min(max(1, n // season_daily), 8)

    # Hard requirement 9: Indexing safety.
    # context[-K * season_daily:] is safe because n >= season_daily and K <= n // season_daily,
    # which implies K * season_daily <= n.
    
    # Extract the last K full daily seasons.
    # 'mat' will have K rows and 'season_daily' columns.
    mat = context[-K * season_daily:].reshape(K, season_daily)

    # Calculate the average seasonal pattern.
    # np.nanmean is used to robustly handle any NaNs present in the input data,
    # ignoring them for the mean calculation. If a column is entirely NaNs,
    # its mean will be NaN. If a column is entirely Infs, its mean will be Inf.
    seasonal_pattern = np.nanmean(mat, axis=0)

    # Tile the derived seasonal pattern to cover the entire prediction_length.
    num_repetitions = int(np.ceil(prediction_length / season_daily))
    forecast_values = np.tile(seasonal_pattern, num_repetitions)[:prediction_length]

    # Hard requirement 4: Never output NaN or inf.
    # Replace any NaNs, positive Infs, or negative Infs in the final forecast
    # with our robustly determined fallback_value.
    return np.nan_to_num(forecast_values, 
                         nan=fallback_value, 
                         posinf=fallback_value, 
                         neginf=fallback_value).astype(float)