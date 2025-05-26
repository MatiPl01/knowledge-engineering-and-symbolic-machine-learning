import numpy as np
from sklearn.covariance import MinCovDet


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    # Fit robust covariance estimator
    mcd = MinCovDet().fit(train_data)
    
    # Mahalanobis distances for training data
    train_mahal = mcd.mahalanobis(train_data)
    threshold = np.max(train_mahal)
    
    # Mahalanobis distances for test data
    test_mahal = mcd.mahalanobis(test_data)
    
    # Anomaly if Mahalanobis distance > threshold
    return (test_mahal > threshold).astype(int)
