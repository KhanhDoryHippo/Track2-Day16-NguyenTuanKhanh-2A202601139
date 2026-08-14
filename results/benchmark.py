#!/usr/bin/env python3
"""LAB 16 - LightGBM Credit Card Fraud Detection Benchmark (CPU)"""
import json, os, time, platform
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

CSV = os.path.expanduser("~/ml-benchmark/creditcard.csv")
OUT = os.path.expanduser("~/ml-benchmark/benchmark_result.json")
SEED = 42

print("=" * 62)
print(" LAB 16 - LightGBM Fraud Detection Benchmark (CPU / t3.medium)")
print("=" * 62)

# ---------- 1. Load data ----------
t0 = time.perf_counter()
df = pd.read_csv(CSV)
load_time = time.perf_counter() - t0
n_fraud = int(df["Class"].sum())
print(f"\n[1] LOAD DATA")
print(f"    Rows x Cols : {df.shape[0]:,} x {df.shape[1]}")
print(f"    Fraud       : {n_fraud:,} ({df['Class'].mean()*100:.3f}%)")
print(f"    Load time   : {load_time:.3f} s")

X = df.drop(columns=["Class"])
y = df["Class"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
X_fit, X_val, y_fit, y_val = train_test_split(
    X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED)

# ---------- 2. Train ----------
n_threads = os.cpu_count()
params = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.05, "num_leaves": 31,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "min_data_in_leaf": 20, "num_threads": n_threads,
    "scale_pos_weight": float((y_fit == 0).sum() / (y_fit == 1).sum()),
    "verbosity": -1, "seed": SEED,
}
dtrain = lgb.Dataset(X_fit, y_fit)
dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

print(f"\n[2] TRAINING  (threads={n_threads}, max 1000 rounds, early stop 50)")
t0 = time.perf_counter()
model = lgb.train(params, dtrain, num_boost_round=1000,
                  valid_sets=[dvalid], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(100)])
train_time = time.perf_counter() - t0
best_iter = model.best_iteration
print(f"    Train time     : {train_time:.3f} s")
print(f"    Best iteration : {best_iter}")

# ---------- 3. Evaluate ----------
y_prob = model.predict(X_te, num_iteration=best_iter)
y_pred = (y_prob >= 0.5).astype(int)
auc  = roc_auc_score(y_te, y_prob)
acc  = accuracy_score(y_te, y_pred)
f1   = f1_score(y_te, y_pred, zero_division=0)
prec = precision_score(y_te, y_pred, zero_division=0)
rec  = recall_score(y_te, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
print(f"\n[3] EVALUATION  (test = {len(y_te):,} rows)")
print(f"    AUC-ROC   : {auc:.6f}")
print(f"    Accuracy  : {acc:.6f}")
print(f"    F1-Score  : {f1:.6f}")
print(f"    Precision : {prec:.6f}")
print(f"    Recall    : {rec:.6f}")
print(f"    Confusion : TN={tn} FP={fp} FN={fn} TP={tp}")

# ---------- 4. Inference latency (1 row) ----------
one = X_te.iloc[[0]]
for _ in range(20):                       # warm-up
    model.predict(one, num_iteration=best_iter)
lat = []
for _ in range(200):
    t = time.perf_counter()
    model.predict(one, num_iteration=best_iter)
    lat.append((time.perf_counter() - t) * 1000.0)
lat = np.array(lat)

# ---------- 5. Throughput (1000 rows) ----------
batch = X_te.iloc[:1000]
for _ in range(3):
    model.predict(batch, num_iteration=best_iter)
runs = []
for _ in range(10):
    t = time.perf_counter()
    model.predict(batch, num_iteration=best_iter)
    runs.append(time.perf_counter() - t)
batch_time = float(np.mean(runs))
throughput = 1000.0 / batch_time

print(f"\n[4] INFERENCE")
print(f"    Latency 1 row   : mean {lat.mean():.3f} ms | p50 {np.percentile(lat,50):.3f} ms | p95 {np.percentile(lat,95):.3f} ms")
print(f"    Batch 1000 rows : {batch_time*1000:.3f} ms  ->  {throughput:,.0f} rows/s")

# ---------- 6. Save ----------
result = {
    "environment": {
        "instance_type": "t3.medium (CPU)",
        "cpu_count": n_threads,
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "platform": platform.platform(),
    },
    "dataset": {
        "rows": int(df.shape[0]), "features": int(X.shape[1]),
        "fraud_count": n_fraud, "fraud_ratio_pct": round(df["Class"].mean()*100, 4),
        "train_rows": int(len(X_fit)), "valid_rows": int(len(X_val)),
        "test_rows": int(len(X_te)),
    },
    "timing": {
        "load_data_sec": round(load_time, 4),
        "training_sec": round(train_time, 4),
        "best_iteration": int(best_iter),
    },
    "metrics": {
        "auc_roc": round(float(auc), 6), "accuracy": round(float(acc), 6),
        "f1_score": round(float(f1), 6), "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    },
    "inference": {
        "latency_1row_ms": {
            "mean": round(float(lat.mean()), 4),
            "p50": round(float(np.percentile(lat, 50)), 4),
            "p95": round(float(np.percentile(lat, 95)), 4),
        },
        "batch_1000rows_ms": round(batch_time * 1000, 4),
        "throughput_rows_per_sec": round(throughput, 2),
    },
}
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n[5] Saved -> {OUT}")
print("=" * 62)
