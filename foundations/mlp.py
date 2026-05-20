import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        n = len(weights)
        h = x @ weights[0] + biases[0]
        if n == 1:
            return np.round(h, 5)

        for i in range(1, n - 1):
            h_next = np.maximum(0, h) @ weights[i] + biases[i]
            h = h_next
            
        o = h @ weights[n - 1] + biases[n - 1]
        return np.round(o, 5)


