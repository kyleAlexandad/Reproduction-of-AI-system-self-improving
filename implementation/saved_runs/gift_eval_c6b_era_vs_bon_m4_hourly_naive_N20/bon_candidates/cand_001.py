import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    """
    Univariate time-series forecasting for hourly data with strong daily seasonality (period 24).

    This function implements a seasonal averaging strategy, taking the last K full
    seasonal cycles (days) and averaging them to derive a robust seasonal pattern.
    It includes robust handling for short context arrays and ensures no NaN/inf outputs.

    Args:
        context (np.array): A 1D numpy array of the past target values for one series.
        prediction_length (int): The forecast horizon.
        freq (str): A frequency string, e.g., "H".
        metadata (dict, optional): An optional dictionary containing additional information
                                   like item_id, season_length, context_length, etc.

    Returns:
        np.array: A 1D numpy array of length `prediction_length` with the point forecasts.
    """
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Handle empty context: return zeros
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Critical benchmark indicates strong daily seasonality for hourly data (period 24)
    season = 24

    # Fallback for short contexts: if not enough data for at least one full season,
    # use the last observed value as a naive forecast.
    if n < season:
        return np.full(prediction_length, float(context[-1]), dtype=float)

    # Determine how many full seasons (days) to average.
    # K is chosen to be at least 1 (if n >= season) and capped at 8 to avoid
    # using too much distant past data, which might not be representative.
    K = min(max(1, n // season), 8)

    # Extract the last K full seasonal cycles from the context.
    # context[-K * season:] safely slices K * season points from the end,
    # as K * season <= n is guaranteed by the 'n < season' check and K calculation.
    recent_seasonal_data = context[-K * season:].reshape(K, season)

    # Calculate the average pattern across these K seasons.
    # This denoises the seasonal pattern by averaging multiple occurrences.
    seasonal_pattern = recent_seasonal_data.mean(axis=0)

    # Tile the derived seasonal pattern to cover the entire prediction_length.
    reps = int(np.ceil(prediction_length / season))
    tiled_forecast = np.tile(seasonal_pattern, reps)[:prediction_length]

    # Ensure no NaN values are returned. If any NaNs occurred during averaging
    # (e.g., due to missing values in the context not handled upstream, though context
    # is usually assumed to be clean), replace them with the last valid context value.
    forecast_values = np.nan_to_num(tiled_forecast, nan=float(context[-1])).astype(float)

    return forecast_values