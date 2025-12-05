# breast_cancer_knn1.py
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---- Load dataset ----
data = load_breast_cancer()
X = data.data          # (569, 30) feature matrix
y = data.target        # (569,) labels

# ---- Train/Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---- Scaling (fit only on train) ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn mean/std & scale
X_test_scaled  = scaler.transform(X_test)        # apply same scaling

print("Train:", X_train_scaled.shape, "Test:", X_test_scaled.shape)
print("hello world")
