import numpy as np


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    # Estimate parameters from training data
    mu = np.mean(train_data)
    sigma = np.std(train_data)
    
    # Calculate z-scores for test data
    z_scores = np.abs((test_data - mu) / sigma)
    
    # Set threshold (3 standard deviations is a common choice)
    threshold = 3.0
    
    # Return 1 for anomalies (z-score > threshold), 0 for normal points
    return (z_scores > threshold).astype(int)
