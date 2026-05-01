#!/usr/bin/env python
"""Train RandomForest classifier to predict best classical scheduler.

Expected input CSV produced by scripts/generate_dataset.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path

# Ensure project root is importable
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

_FEATURES = [
    "n_proc",
    "avg_burst",
    "std_burst",
    "avg_arrival_gap",
]
_LABEL = "best_algo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI scheduler model")
    parser.add_argument("dataset", type=Path, help="CSV dataset path")
    parser.add_argument("--model_out", type=Path, default=Path("data/model.pkl"))
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.dataset)

    X = df[_FEATURES].values
    y = df[_LABEL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
