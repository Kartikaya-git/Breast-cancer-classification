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

# part3_train_time.py
from sklearn.neighbors import KNeighborsClassifier

# ------- Config -------
K = 5  # number of neighbors

# ------- Build model -------
knn = KNeighborsClassifier(
    n_neighbors=K,        # K value
    metric='minkowski',   # distance type (Euclidean when p=2)
    p=2,
    weights='uniform',    # 'distance' to weight closer neighbors more
    n_jobs=-1             # use all CPU cores for predict
)

# ------- Train (fit) -------
t0 = time.perf_counter()
knn.fit(X_train_scaled, y_train)   # stores training data (cheap)
train_time = time.perf_counter() - t0

# ------- Predict (test) -------
t0 = time.perf_counter()
y_pred = knn.predict(X_test_scaled)     # nearest-neighbor search happens here
predict_time = time.perf_counter() - t0

# ------- Probabilities for AUC (optional) -------
t0 = time.perf_counter()
y_proba = knn.predict_proba(X_test_scaled)[:, 1]   # returns class probabilities
proba_time = time.perf_counter() - t0

# ------- Per-sample latency -------
n_test = X_test_scaled.shape[0]
per_sample_ms = (predict_time / n_test) * 1000

# ------- Print results -------
print(f"K = {K}")
print(f"Train time        : {train_time:.6f} s")             # time to fit (store data)
print(f"Predict time      : {predict_time:.6f} s")             # time to predict class labels
print(f"Predict time/prob.: {proba_time:.6f} s")               # time to compute probabilities
print(f"Per-sample latency: {per_sample_ms:.4f} ms ({n_test} samples)")


