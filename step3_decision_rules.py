"""
STEP 3 (complete): Four-outcome cleaning-verification decision rules.
Outcomes: sufficient / more_cleaning / unclear_image / expert_inspection

Pipeline per the team spec:
  1. Photo-quality check on the AFTER photo (sharpness, brightness, contrast)
       -> any failure = unclear_image
  2. Fouling score = fraction of pixels whose CIELAB colour distance from the
     clean reference colour exceeds a colour threshold
  3. Zone-specific rules -> sufficient / more_cleaning / expert_inspection
  4. Naive baseline to beat (one fixed threshold on the after-score)
  5. Optional logistic-regression classifier on the same features
  6. Thresholds are TUNED ON THE TUNE SPLIT ONLY, then run ONCE on the test split

Run:
  python3 step3_decision_rules.py dataset/

Honesty notes:
  - On our synthetic dataset there is no separate clean-reference photo, so the
    clean paint colour is estimated as the median Lab colour of the image
    (valid while fouling covers < ~50% of the area). If you have a real clean
    reference photo, pass its median instead - same code path.
  - TEST-split numbers are the only ones to quote. Tune numbers are a sanity check.

Needs: python3, opencv-python (cv2), numpy; scikit-learn optional (for step 5).
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SENSITIVE_ZONES = {"sea_chest", "propeller", "anodes"}
LABELS = ["sufficient", "more_cleaning", "unclear_image", "expert_inspection"]

# Constants (document any change)
# LAB_COLOUR_THRESHOLD chosen on the TUNE split: at 25 residual speckle/JPEG noise
# in fully-cleaned images still counted as fouling (sufficient .085-.167 overlapped
# more_cleaning .124-.254); at 45 the groups separate (.001-.009 vs .026-.056).
LAB_COLOUR_THRESHOLD = 45.0   # Lab distance above which a pixel counts as "fouled"
T_BRIGHT_HI = 245.0           # mean grey above this = overexposed (no tune examples; engineering choice)


# ---------------------------------------------------------------- features
def read_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def quality_features(path):
    """Sharpness (variance of Laplacian), brightness (mean grey), contrast (std grey)."""
    grey = cv2.cvtColor(read_bgr(path), cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": float(cv2.Laplacian(grey, cv2.CV_64F).var()),
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
    }


def fouling_score(path, clean_lab=None):
    """Fraction of pixels whose Lab distance from the clean paint colour > threshold.
    clean_lab: Lab colour of the clean surface. If None, estimated as the image's
    per-channel median in Lab space (stand-in for a clean reference photo)."""
    lab = cv2.cvtColor(read_bgr(path), cv2.COLOR_BGR2LAB).astype(np.float32)
    if clean_lab is None:
        clean_lab = np.median(lab.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(lab - clean_lab, axis=-1)
    return float((dist > LAB_COLOUR_THRESHOLD).mean())


def pair_features(root, row):
    q = quality_features(root / row["after_file"])
    s_before = fouling_score(root / row["before_file"])
    s_after = fouling_score(root / row["after_file"])
    return {**q, "s_before": s_before, "s_after": s_after,
            "improvement": s_before - s_after,
            "sensitive": 1.0 if row["zone"] in SENSITIVE_ZONES else 0.0}


# ---------------------------------------------------------------- tuning (step 6)
def midpoint(lo_side_max, hi_side_min, fallback):
    """Threshold halfway between two separated groups; fallback if they overlap."""
    if lo_side_max < hi_side_min:
        return (lo_side_max + hi_side_min) / 2.0
    return fallback


def tune_thresholds(feats_tune):
    """Choose every threshold from the TUNE split only. Prints its reasoning."""
    f = feats_tune
    unclear = [x for x in f if x["label"] == "unclear_image"]
    clear = [x for x in f if x["label"] != "unclear_image"]
    suff = [x for x in f if x["label"] == "sufficient"]
    more = [x for x in f if x["label"] == "more_cleaning"]

    th = {}
    th["T_SHARP"] = midpoint(max(x["sharpness"] for x in unclear),
                             min(x["sharpness"] for x in clear), fallback=100.0)
    dark = [x for x in unclear if x["brightness"] < 40]
    th["T_BRIGHT_LO"] = midpoint(max((x["brightness"] for x in dark), default=15.0),
                                 min(x["brightness"] for x in clear), fallback=30.0)
    # contrast: only fires on extreme haze; set well below any clear image
    th["T_CONTRAST"] = 0.5 * min(x["contrast"] for x in clear)
    t_clean = midpoint(max(x["s_after"] for x in suff),
                       min(x["s_after"] for x in more), fallback=0.05)
    th["T_CLEAN"] = {"default": t_clean, "sensitive": 0.8 * t_clean}
    th["T_IMPROVE"] = 0.5 * min(x["improvement"] for x in more)   # below = "cleaning did nothing"
    th["T_SEVERE"] = min(1.0, 1.5 * max(x["s_after"] for x in more))
    th["T_BASELINE"] = t_clean    # the naive baseline gets the same tuned number, to be fair

    print("Thresholds tuned on the TUNE split:")
    for k, v in th.items():
        print(f"  {k} = {v if not isinstance(v, dict) else {z: round(x,4) for z,x in v.items()}}"
              if isinstance(v, dict) else f"  {k} = {v:.4f}")
    return th


# ---------------------------------------------------------------- rules (steps 1-3)
def decide(feat, zone, th):
    # Step 1: photo quality on the after photo
    if (feat["sharpness"] < th["T_SHARP"]
            or feat["brightness"] < th["T_BRIGHT_LO"] or feat["brightness"] > T_BRIGHT_HI
            or feat["contrast"] < th["T_CONTRAST"]):
        return "unclear_image"
    # Steps 2-3: fouling score + zone rules
    t_clean = th["T_CLEAN"]["sensitive" if zone in SENSITIVE_ZONES else "default"]
    if feat["s_after"] <= t_clean:
        return "sufficient"
    if (feat["s_after"] > th["T_SEVERE"]
            or feat["improvement"] < th["T_IMPROVE"]
            or zone in SENSITIVE_ZONES):
        return "expert_inspection"
    return "more_cleaning"


# Step 4: baseline to beat
def decide_baseline(feat, zone, th):
    return "sufficient" if feat["s_after"] <= th["T_BASELINE"] else "more_cleaning"


# ---------------------------------------------------------------- evaluation
def evaluate(feats, decide_fn, th, name):
    cm = {t: {p: 0 for p in LABELS} for t in LABELS}
    out_rows = []
    for x in feats:
        pred = decide_fn(x, x["zone"], th)
        cm[x["label"]][pred] += 1
        out_rows.append({**x, "pred": pred})
    n = len(feats)
    acc = sum(cm[t][t] for t in LABELS) / n
    dirty = [x for x in out_rows if x["label"] in ("more_cleaning", "expert_inspection")]
    false_suff = sum(1 for x in dirty if x["pred"] == "sufficient")
    print(f"\n== {name} ==  n={n}  accuracy={acc:.0%}  "
          f"dirty-passed-as-clean={false_suff}/{len(dirty)}")
    print("   " + " ".join(f"{l[:9]:>9}" for l in ["truth\\pred"] + LABELS))
    for t in LABELS:
        print(f"   {t[:9]:>10} " + " ".join(f"{cm[t][p]:>9}" for p in LABELS))
    return {"name": name, "n": n, "accuracy": acc,
            "false_sufficient": false_suff, "dirty_total": len(dirty),
            "confusion": cm, "rows": out_rows}


# ---------------------------------------------------------------- step 5 (optional)
def logistic_regression(feats_tune, feats_test):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except ImportError:
        print("\n(step 5 skipped: scikit-learn not installed)")
        return None
    cols = ["sharpness", "brightness", "contrast", "s_before", "s_after",
            "improvement", "sensitive"]
    Xtr = np.array([[x[c] for c in cols] for x in feats_tune])
    ytr = [x["label"] for x in feats_tune]
    Xte = np.array([[x[c] for c in cols] for x in feats_test])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    feats_pred = [dict(x) for x in feats_test]
    for x, p in zip(feats_pred, preds):
        x["_lr"] = p
    return evaluate(feats_pred, lambda f, z, th: f["_lr"], None,
                    "logistic regression on TEST split (step 5)")


# ---------------------------------------------------------------- main
def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
    rows = list(csv.DictReader(open(root / "labels.csv")))
    print(f"Computing features for {len(rows)} pairs...")
    feats = [{**r, **pair_features(root, r)} for r in rows]
    tune = [x for x in feats if x["split"] == "tune"]
    test = [x for x in feats if x["split"] == "test"]

    th = tune_thresholds(tune)
    evaluate(tune, decide, th, "rules on TUNE split (sanity check only)")
    res_rules = evaluate(test, decide, th, "rules on TEST split  <-- quote these")
    res_base = evaluate(test, decide_baseline, th, "naive baseline on TEST split")
    res_lr = logistic_regression(tune, test)

    out = {"thresholds": {k: v for k, v in th.items()},
           "rules_test": {k: res_rules[k] for k in ("n", "accuracy", "false_sufficient",
                                                    "dirty_total", "confusion")},
           "baseline_test": {k: res_base[k] for k in ("n", "accuracy", "false_sufficient",
                                                      "dirty_total", "confusion")},
           "per_pair": [{k: v for k, v in x.items()} for x in res_rules["rows"]]}
    if res_lr:
        out["logreg_test"] = {k: res_lr[k] for k in ("n", "accuracy", "false_sufficient",
                                                     "dirty_total", "confusion")}
    with open(root.parent / "results_step3.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nWrote {root.parent / 'results_step3.json'} (feed this to the 3D view / dashboard)")


if __name__ == "__main__":
    main()
