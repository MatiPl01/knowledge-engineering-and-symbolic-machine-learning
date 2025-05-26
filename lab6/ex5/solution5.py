import numpy as np


def reconstruction_errors(inputs: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """Calculate mean squared error for each sample."""
    return np.mean((inputs - reconstructions) ** 2, axis=tuple(range(1, inputs.ndim)))


def calc_threshold(reconstr_err_nominal: np.ndarray) -> float:
    """Set threshold as 99th percentile of nominal errors."""
    return np.percentile(reconstr_err_nominal, 99)


def detect(reconstr_err_all: np.ndarray, threshold: float) -> list:
    """Classify as anomaly if error > threshold."""
    return (reconstr_err_all > threshold).astype(int).tolist()
