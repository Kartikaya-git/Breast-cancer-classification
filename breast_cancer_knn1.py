# part1_load.py
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = load_breast_cancer()
X = data.data      # shape (569, 30)
y = data.target    # 0 = benign, 1 = malignant

print("Data shape:", X.shape, "Labels shape:", y.shape)
print("Class distribution:", np.bincount(y))
