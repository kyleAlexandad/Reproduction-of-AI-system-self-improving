import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

def train_and_predict(train_path, test_path):
    """
    Trains an improved binary classification model using HistGradientBoostingClassifier
    with feature scaling and predicts probabilities for the test set.

    Args:
        train_path (str): Path to the training CSV file containing features and 'target'.
        test_path (str): Path to the test CSV file containing only features.

    Returns:
        np.ndarray: Predicted probabilities for the positive class (class 1) on the test data.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Separate features and target from the training data
    X_train = train.drop('target', axis=1)
    y_train = train['target']

    # Test data contains features only, make a copy to ensure no modifications to original test DataFrame
    X_test = test.copy()

    # --- Feature Engineering / Scaling ---
    # Standardize features using StandardScaler.
    # It's important to fit the scaler only on the training data
    # and then transform both training and test data using the fitted scaler.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Model Definition and Training ---
    # Using HistGradientBoostingClassifier, a fast and efficient gradient boosting model
    # that often outperforms RandomForestClassifier and traditional GradientBoostingClassifier
    # for a given number of estimators/iterations, especially when speed is critical.
    # max_iter is analogous to n_estimators, set to 50 as per the constraint.
    # Hyperparameters (learning_rate, max_depth) are manually tuned for potentially
    # better performance within the limited number of iterations, aiming for a balance
    # between learning speed and generalization.
    model = HistGradientBoostingClassifier(
        max_iter=50,             # Equivalent to n_estimators, capped at 50 for performance
        learning_rate=0.15,      # Slightly increased from default 0.1 for faster convergence with limited iterations
        max_depth=5,             # Limit tree depth to prevent overfitting and encourage ensemble learning
        random_state=42,         # For reproducibility
        # HistGradientBoostingClassifier is inherently optimized for speed and parallel execution
        # and doesn't require a separate 'n_jobs' parameter like RandomForest.
    )

    # Train the model on the scaled training data
    model.fit(X_train_scaled, y_train)

    # --- Prediction ---
    # Return probability of the positive class (class 1) for ROC-AUC evaluation.
    # predict_proba returns probabilities for both classes [P(class 0), P(class 1)].
    # We need the probabilities for class 1, which is the second column (index 1).
    return model.predict_proba(X_test_scaled)[:, 1]