# 2) Dobór parametrów:
# - EllipticEnvelope: contamination – oczekiwany udział anomalii (wpływa na czułość i precyzję).
# - One-Class SVM: nu – górna granica udziału anomalii; kernel i gamma – kształt granicy decyzyjnej.
# - Isolation Forest: contamination – oczekiwany udział anomalii; n_estimators – liczba drzew (wpływa na stabilność).
# - Local Outlier Factor: contamination – oczekiwany udział anomalii; n_neighbors – liczba sąsiadów (wpływa na lokalność detekcji).
#
# 3) Wyniki na zbiorze uczącym:
# Covariance-Mahalanobis: F1-score = 0.89, macierz pomyłek: [[8885  115], [ 115  885]]
# One-Class SVM:         F1-score = 0.94, macierz pomyłek: [[8939   61], [  62  938]]
# Isolation Forest:      F1-score = 0.96, macierz pomyłek: [[8956   44], [  44  956]]
# Local Outlier Factor:  F1-score = 0.23, macierz pomyłek: [[8232  768], [ 768  232]]
#
# 4) Porównanie i skłonności metod:
# - Covariance-Mahalanobis: dobrze działa dla rozkładów zbliżonych do Gaussa, umiarkowana czułość.
# - One-Class SVM: uniwersalny, dobrze radzi sobie z danymi o złożonej strukturze, wymaga strojenia parametrów.
# - Isolation Forest: bardzo skuteczny dla dużych i nietypowych zbiorów, odporny na różne rozkłady.
# - Local Outlier Factor: wykrywa lokalne anomalie, ale w tym zadaniu ma niską skuteczność (może wymagać strojenia n_neighbors).

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from utils import binary2neg_boolean
import numpy as np

SEED = 1


def detect_cov(data: np.ndarray, outliers_fraction: float) -> list:
    model = EllipticEnvelope(contamination=outliers_fraction, random_state=SEED)
    pred = model.fit_predict(data)
    return binary2neg_boolean(pred)


def detect_ocsvm(data: np.ndarray, outliers_fraction: float) -> list:
    model = svm.OneClassSVM(nu=outliers_fraction, kernel='rbf', gamma='scale')
    model.fit(data)
    pred = model.predict(data)
    return binary2neg_boolean(pred)


def detect_iforest(data: np.ndarray, outliers_fraction: float) -> list:
    model = IsolationForest(contamination=outliers_fraction, random_state=SEED)
    pred = model.fit_predict(data)
    return binary2neg_boolean(pred)


def detect_lof(data: np.ndarray, outliers_fraction: float) -> list:
    model = LocalOutlierFactor(contamination=outliers_fraction)
    pred = model.fit_predict(data)
    return binary2neg_boolean(pred)
