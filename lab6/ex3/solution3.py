# 3) Komentarz:
# Mahalanobis jest bardzo konserwatywny – wykrywa tylko punkty znacznie odbiegające od rozkładu uczącego (mało fałszywych alarmów, ale więcej pominiętych anomalii).
# OC-SVM jest bardziej elastyczny – wykrywa więcej anomalii (wyższa czułość), ale kosztem większej liczby fałszywych alarmów.
# Mahalanobis sprawdza się, gdy dane są dobrze opisane pojedynczym rozkładem Gaussa i zależy nam na niskiej liczbie fałszywych alarmów.
# OC-SVM lepiej radzi sobie z danymi o złożonej strukturze lub wielomodalnymi.
#
# 4) Wpływ parametrów OC-SVM:
# Zwiększenie nu powoduje, że algorytm wykrywa więcej anomalii (więcej fałszywych alarmów).
# Zmiana gamma wpływa na kształt granicy decyzyjnej – większe gamma daje bardziej złożoną granicę.
# Wybór kernela zmienia sposób separacji anomalii od danych normalnych.
from sklearn import svm
from utils import binary2neg_boolean
import numpy as np


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    # OneClass-SVM
    clf = svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    clf.fit(train_data)
    svm_pred = binary2neg_boolean(clf.predict(test_data))
    return svm_pred
