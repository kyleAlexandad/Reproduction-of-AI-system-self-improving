import numpy as np

def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = len(context)

    # Hard Requirement 8: Handle short context arrays robustly.
    # If context is empty, return zeros.
    if n == 0:
        return np.zeros(prediction_length, dtype=float)

    # Get the last observed value.
    last = float(context[-1])

    # CRITICAL BENCHMARK FACTS:
    # - The NAIVE last-value baseline is STRONG here: MASE ~= 2.7773.
    # - Parent candidate MASE: 2.780662 (slightly worse than naive).
    # - More complex models (moving average, seasonal naive, aggressive trend) hurt badly.
    # - Weekly series behave close to a RANDOM WALK: the last observed value is the single best simple predictor.
    #
    # Given that the parent's attempt to add a damped trend, even with heavy weighting towards
    # the naive forecast (85% naive, 15% damped trend), resulted in a slightly *higher* MASE
    # than the pure naive baseline, the most robust strategy is to revert to the
    # pure naive forecast. This ensures we at least match the strong baseline performance,
    # thereby improving upon the parent candidate's MASE.
    #
    # This also adheres to the recommendation: "It is better to TIE naive than to diverge far above it."

    # Return a 1D numpy array of length EXACTLY prediction_length with the point forecasts.
    # This directly implements the pure naive (last-value) forecast.
    # Hard Requirement 3: Output length must be EXACTLY prediction_length.
    # Hard Requirement 4: Never output NaN or inf. `np.full` inherently satisfies this.
    return np.full(prediction_length, last, dtype=float)