"""
Four-outcome cleaning-verification decision rules + evaluation.
Outcomes: sufficient / more_cleaning / unclear_image / expert_inspection

Run:  python3 classify.py dataset/
Tunes nothing at runtime -- thresholds below were chosen on the 'tune' split only.
Reports results on the held-out 'test' split, plus a simple baseline comparison.
Writes results.json next to the dataset for the dashboard / 3D view.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SENSITIVE_ZONES = {"sea_chest", "propeller", "anodes"}

# ---- thresholds (chosen on the tune split; document any change you make) ----
# Tune-split feature ranges that set them:
#   sharpness: unclear 1.6-85 vs clear 1075+  -> T_SHARP = 200
#   brightness: dark-unclear 14-20 vs clear 69+ -> T_BRIGHT_LO = 30
#   after-score: sufficient .019-.048 vs more_cleaning .051-.107 -> T_CLEAN = .050
T_SHARP = 200.0    # Laplacian variance below this => blurry
T_BRIGHT_LO = 30.0  # mean brightness below this => too dark
T_CLEAN = {"default": 0.050, "sensitive": 0.040}  # fouling score <= this => clean
T_IMPROVE = 0.005   # improvement below this (while dirty) => expert ("cleaning did nothing")
T_SEVERE = 0.45     # after-score above this => expert


def load_grey(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def laplacian_var(grey):
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(grey, (3, 3))
    lap = (w * k).sum(axis=(-1, -2))
    return float(lap.var())


def quality(path):
    g = load_grey(path)
    small = g[::2, ::2]
    return {
        "sharpness": laplacian_var(small),
        "brightness": float(g.mean()),
        "contrast": float(g.std()),
    }


def fouling_score(path):
    """Fraction of pixels far from the estimated clean paint colour.
    The clean colour is the per-channel median -- works when fouling < ~50%."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    med = np.median(arr.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(arr - med, axis=-1)
    return float((dist > 55.0).mean())


def decide(before_path, after_path, zone):
    q = quality(after_path)
    if q["sharpness"] < T_SHARP or q["brightness"] < T_BRIGHT_LO:
        return "unclear_image", {"quality": q}

    s_before = fouling_score(before_path)
    s_after = fouling_score(after_path)
    improvement = s_before - s_after
    t_clean = T_CLEAN["sensitive"] if zone in SENSITIVE_ZONES else T_CLEAN["default"]
    detail = {"quality": q, "score_before": round(s_before, 3),
              "score_after": round(s_after, 3), "improvement": round(improvement, 3)}

    if s_after <= t_clean:
        return "sufficient", detail
    if s_after > T_SEVERE or improvement < T_IMPROVE or zone in SENSITIVE_ZONES:
        return "expert_inspection", detail
    return "more_cleaning", detail


def baseline(before_path, after_path, zone):
    """Naive baseline: one fixed threshold on the after image, nothing else."""
    return "sufficient" if fouling_score(after_path) <= 0.08 else "more_cleaning"


def evaluate(rows, root, fn, name):
    labels = ["sufficient", "more_cleaning", "unclear_image", "expert_inspection"]
    cm = {t: {p: 0 for p in labels} for t in labels}
    details = []
    for r in rows:
        res = fn(root / r["before_file"], root / r["after_file"], r["zone"])
        pred = res[0] if isinstance(res, tuple) else res
        det = res[1] if isinstance(res, tuple) else {}
        cm[r["label"]][pred] += 1
        details.append({**r, "pred": pred, **det})
    n = len(rows)
    correct = sum(cm[t][t] for t in labels)
    truly_dirty = [d for d in details if d["label"] in ("more_cleaning", "expert_inspection")]
    false_suff = sum(1 for d in truly_dirty if d["pred"] == "sufficient")
    print(f"\n== {name} ==  n={n}  accuracy={correct}/{n} = {correct/n:.0%}")
    print(f"   wrongly-called-sufficient (dirty areas passed as clean): "
          f"{false_suff}/{len(truly_dirty)}" if truly_dirty else "   (no dirty areas)")
    print("   confusion (rows=truth, cols=prediction):")
    print("   " + " ".join(f"{l[:9]:>9}" for l in [""] + labels))
    for t in labels:
        print(f"   {t[:9]:>9} " + " ".join(f"{cm[t][p]:>9}" for p in labels))
    return {"name": name, "n": n, "accuracy": correct / n,
            "false_sufficient": false_suff, "dirty_total": len(truly_dirty),
            "confusion": cm, "details": details}


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
    rows = list(csv.DictReader(open(root / "labels.csv")))
    tune = [r for r in rows if r["split"] == "tune"]
    test = [r for r in rows if r["split"] == "test"]

    print("Thresholds were tuned on the TUNE split only; TEST numbers are the honest ones.")
    evaluate(tune, root, decide, "rules on TUNE split (sanity check)")
    res_rules = evaluate(test, root, decide, "rules on TEST split")
    res_base = evaluate(test, root, baseline, "naive baseline on TEST split")

    out = {"rules_test": {k: res_rules[k] for k in ("n", "accuracy", "false_sufficient", "dirty_total", "confusion")},
           "baseline_test": {k: res_base[k] for k in ("n", "accuracy", "false_sufficient", "dirty_total", "confusion")},
           "per_pair": res_rules["details"]}
    with open(root.parent / "results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWrote {root.parent / 'results.json'}")


if __name__ == "__main__":
    main()
