import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = y_true.size
        epsilon = 10**(-7)
        result = 0
        for i in range(n):
            result += ((y_true[i] * np.log(y_pred[i] + epsilon)) + ((1 - y_true[i]) * math.log(1 - y_pred[i] + epsilon)))
        result /= (-n)
        return np.round(result, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 10**-7
        result = 0
        for i in range(y_true.shape[0]):
            for c in range(y_true.shape[1]):
                result += y_true[i, c] * np.log(y_pred[i, c] + epsilon)
        result /= -y_true.shape[0]
        return np.round(result, 4)
