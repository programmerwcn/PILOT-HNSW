#!/usr/bin/env python3
"""Train SIFT100M effort tiers from the 59-feature zero-anchor probe recipe."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ("easy", "mid", "hard")
BUG_COMPAT_META = tuple(range(16, 24))


def load_features(path):
    payload = json.load(path.open())
    names = payload["feature_names"]
    rows = payload["queries"]
    qids = [int(row["query_index"]) for row in rows]
    if qids != list(range(len(rows))):
        raise ValueError("features must cover contiguous query ids from zero")
    x = np.asarray([row["features"] for row in rows], np.float32)
    if x.shape[1] != 59 or len(names) != 59:
        raise ValueError(f"expected 59 features, got {x.shape} / {len(names)} names")
    if not np.isfinite(x).all():
        raise ValueError("non-finite feature")
    if payload["metadata"].get("anchor_feature_count") != 0:
        raise ValueError("anchor-dependent features are forbidden")
    return x, names, payload["metadata"]


def fit_variant(name, x, feature_names, y, ntrain, ntest, threads, out_dir):
    model = LGBMClassifier(
        objective="multiclass", n_estimators=400, learning_rate=0.05,
        num_leaves=31, random_state=42, verbose=-1, n_jobs=threads,
    )
    model.fit(x[:ntrain], y[:ntrain], feature_name=feature_names)
    pred = model.predict(x[ntrain:ntrain + ntest]).astype(int)
    prob = model.predict_proba(x[ntrain:ntrain + ntest])
    truth = y[ntrain:ntrain + ntest]
    model_path = out_dir / f"zero_anchor_{name}_lgbm.joblib"
    text_path = out_dir / f"zero_anchor_{name}_lgbm.txt"
    joblib.dump({"model": model, "feature_names": feature_names}, model_path)
    model.booster_.save_model(str(text_path))
    return {
        "name": name,
        "feature_count": x.shape[1],
        "feature_names": feature_names,
        "accuracy": float(np.mean(pred == truth)),
        "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            truth, pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0),
        "predicted_tiers": pred.tolist(),
        "probabilities": prob.tolist(),
        "model": str(model_path),
        "native_model": str(text_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--train-queries", type=int, default=5000)
    ap.add_argument("--test-queries", type=int, default=1000)
    ap.add_argument("--fixed-ef", type=int, default=500)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    x59, names59, feature_meta = load_features(args.features)
    labels = json.load(args.labels.open())["queries"]
    needed = args.train_queries + args.test_queries
    if len(x59) < needed or len(labels) < needed:
        raise ValueError("features/labels do not cover train+test")
    ef = np.asarray([row["min_ef"] for row in labels[:needed]], np.float64)
    # The requested tiering rule: sort min_ef from low to high, then split the
    # training population into equal thirds (easy, medium, hard).
    order = np.argsort(ef[:args.train_queries], kind="stable")
    y = np.empty(needed, np.int64)
    train_groups = np.array_split(order, 3)
    for tier, ids in enumerate(train_groups):
        y[ids] = tier
    # Held-out tiers use the maximum min_ef in the first two training thirds.
    easy_max = int(np.max(ef[train_groups[0]]))
    medium_max = int(np.max(ef[train_groups[1]]))
    test_ef = ef[args.train_queries:needed]
    y[args.train_queries:needed] = np.where(
        test_ef <= easy_max, 0,
        np.where(test_ef <= medium_max, 1, 2))
    caps = [int(np.max(ef[ids])) for ids in train_groups]
    constrained_caps = [min(args.fixed_ef, cap) for cap in caps]

    keep51 = [i for i in range(59) if i not in BUG_COMPAT_META]
    x51 = x59[:, keep51]
    names51 = [names59[i] for i in keep51]
    args.model_dir.mkdir(parents=True, exist_ok=True)
    variants = {
        "compatible59": fit_variant("compatible59", x59, names59, y,
                                     args.train_queries, args.test_queries,
                                     args.threads, args.model_dir),
        "clean51": fit_variant("clean51", x51, names51, y,
                                args.train_queries, args.test_queries,
                                args.threads, args.model_dir),
    }
    # Serving defaults to the clean model unless the compatibility fields prove
    # materially useful on SIFT. The threshold is fixed before examining output.
    selected = "compatible59" if (
        variants["compatible59"]["accuracy"] - variants["clean51"]["accuracy"] > 0.002
    ) else "clean51"
    result = {
        "recipe": "zero-anchor common in-graph probe",
        "feature_metadata": feature_meta,
        "tiering_rule": "ascending min_ef, equal training thirds",
        "tier_min_ef_boundaries": {"easy_max": easy_max,
                                   "medium_max": medium_max},
        "tier_ef_initial": caps,
        "tier_ef_fixed500_constrained": constrained_caps,
        "adaptive_exceeds_fixed": max(caps) > args.fixed_ef,
        "fixed_reference_ef": args.fixed_ef, "k": args.k,
        "train_query_range": [0, args.train_queries],
        "test_query_range": [args.train_queries, needed],
        "query_ids": list(range(args.train_queries, needed)),
        "true_tiers": y[args.train_queries:needed].astype(int).tolist(),
        "variants": variants, "selected_variant": selected,
        "predicted_tiers": variants[selected]["predicted_tiers"],
        "probabilities": variants[selected]["probabilities"],
        "accuracy": variants[selected]["accuracy"],
        "confusion_matrix": variants[selected]["confusion_matrix"],
        "monitor_cost": {"num_anchors": 0,
                         "full_distance_equivalents_per_query": 0,
                         "mean_seconds_per_query": 0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    print(f"[model] min_ef boundaries={[easy_max, medium_max]} caps={caps} constrained={constrained_caps}")
    for name, row in variants.items():
        print(f"[{name}] features={row['feature_count']} accuracy={row['accuracy']:.4f}")
        print(np.asarray(row["confusion_matrix"]))
    print(f"[selected] {selected}\n[save] {args.output}")


if __name__ == "__main__":
    main()
