"""
STEP 4: Monte-Carlo simulation of three hull-cleaning strategies.

  A: repeat the full route (two full cleaning passes, no verification)
  B: clean once and trust it
  C: verify-then-revisit (ours) -- AI decisions drawn from the MEASURED
     confusion matrix of step3_decision_rules.py (results_step3.json),
     so the simulation inherits the real, tested error rates.

Outputs COUNTS, not money (area-visits, checks, re-photos, revisits,
expert areas, dirty areas left, wrongly-"clean" areas). The web page
multiplies counts by editable cost inputs, so sliders recompute instantly.

ALL zone probabilities below are ASSUMPTIONS for the demo - say so.
Context only (not our result): IMO GreenVoyage2050 lists a typical full
in-water cleaning at $5,000-$50,000.

Run:  python3 step4_simulation.py
Needs results_step3.json (from step3) and dataset/labels.csv in this folder.
Writes: sim_results.json and sim_results.js (same data; the .js loads from file://).
"""
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
N_HULLS = 1000
rng = random.Random(7)

# ---------------------------------------------------------------- hull layout
# 20 columns along the length (col 0 = stern, col 19 = bow) x 10 rows around
# the girth (row 0 = port waterline, rows 4-5 = flat bottom, row 9 = stbd waterline).
COLS, ROWS = 20, 10
ROW_ZONE = {0: "waterline", 1: "vertical_side", 2: "vertical_side", 3: "bilge_keel",
            4: "flat_bottom", 5: "flat_bottom", 6: "bilge_keel", 7: "vertical_side",
            8: "vertical_side", 9: "waterline"}
SPECIAL = {(0, 3): "propeller", (0, 4): "propeller", (0, 5): "propeller", (0, 6): "propeller",
           (6, 2): "sea_chest", (13, 7): "sea_chest",
           (4, 6): "anodes", (10, 3): "anodes", (16, 6): "anodes"}
SENSITIVE = {"sea_chest", "propeller", "anodes"}

# -------------------------------------------------- ASSUMPTIONS (label them!)
# p_foul: chance the area is fouled before cleaning
# p_clean1: chance one cleaning pass actually cleans it (lower in niche zones)
# p_clean2: chance a TARGETED revisit cleans it
# p_unclear: chance the verification photo is unusable (murk, shadow, angle)
ZONE_ASSUMPTIONS = {
    "flat_bottom":  {"p_foul": 0.50, "p_clean1": 0.85, "p_clean2": 0.95, "p_unclear": 0.08},
    "vertical_side": {"p_foul": 0.45, "p_clean1": 0.85, "p_clean2": 0.95, "p_unclear": 0.08},
    "waterline":    {"p_foul": 0.60, "p_clean1": 0.80, "p_clean2": 0.95, "p_unclear": 0.10},
    "bow":          {"p_foul": 0.40, "p_clean1": 0.80, "p_clean2": 0.95, "p_unclear": 0.08},
    "stern":        {"p_foul": 0.55, "p_clean1": 0.75, "p_clean2": 0.90, "p_unclear": 0.12},
    "bilge_keel":   {"p_foul": 0.65, "p_clean1": 0.60, "p_clean2": 0.85, "p_unclear": 0.12},
    "sea_chest":    {"p_foul": 0.80, "p_clean1": 0.45, "p_clean2": 0.75, "p_unclear": 0.15},
    "propeller":    {"p_foul": 0.75, "p_clean1": 0.50, "p_clean2": 0.80, "p_unclear": 0.15},
    "anodes":       {"p_foul": 0.70, "p_clean1": 0.50, "p_clean2": 0.80, "p_unclear": 0.12},
}


def build_areas():
    areas, order = [], 0
    for c in range(COLS):                       # serpentine robot route
        rows = range(ROWS) if c % 2 == 0 else range(ROWS - 1, -1, -1)
        for r in rows:
            zone = SPECIAL.get((c, r), "bow" if c == COLS - 1 else ROW_ZONE[r])
            areas.append({"id": f"a{c:02d}{r}", "col": c, "row": r,
                          "zone": zone, "order": order})
            order += 1
    return areas


# ------------------------------------------------ measured confusion matrix
def load_confusion():
    data = json.load(open(HERE / "results_step3.json"))
    cm = data["rules_test"]["confusion"]
    probs = {}
    for truth, row in cm.items():
        n = sum(row.values())
        probs[truth] = [(pred, cnt / n) for pred, cnt in row.items() if cnt > 0] if n else [(truth, 1.0)]
    return cm, probs


def sample_pred(probs, truth):
    u, acc = rng.random(), 0.0
    for pred, p in probs[truth]:
        acc += p
        if u <= acc:
            return pred
    return probs[truth][-1][0]


def truth_label(area, dirty, photo_unclear):
    if photo_unclear:
        return "unclear_image"
    if not dirty:
        return "sufficient"
    return "expert_inspection" if area["zone"] in SENSITIVE else "more_cleaning"


# ---------------------------------------------------------------- one hull
def simulate_hull(areas, probs, record_demo=False):
    st = {k: {"visits": 0, "checks": 0, "rephotos": 0, "revisits": 0,
              "expert_areas": 0, "dirty_unflagged": 0, "wrongly_clean": 0}
          for k in "ABC"}
    demo = []
    for a in areas:
        z = ZONE_ASSUMPTIONS[a["zone"]]
        fouled = rng.random() < z["p_foul"]
        cleaned1 = (not fouled) or (rng.random() < z["p_clean1"])
        dirty1 = fouled and not cleaned1

        # A: two blind full passes
        st["A"]["visits"] += 2
        dirtyA = dirty1 and not (rng.random() < z["p_clean1"])
        st["A"]["dirty_unflagged"] += dirtyA
        st["A"]["wrongly_clean"] += dirtyA          # assumed clean, never verified

        # B: one blind pass
        st["B"]["visits"] += 1
        st["B"]["dirty_unflagged"] += dirty1
        st["B"]["wrongly_clean"] += dirty1

        # C: one pass + verify every area
        st["C"]["visits"] += 1
        st["C"]["checks"] += 1
        unclear = rng.random() < z["p_unclear"]
        truth = truth_label(a, dirty1, unclear)
        pred = sample_pred(probs, truth)
        pred1, rephoto = pred, False
        if pred == "unclear_image":                 # one re-photo, then re-decide
            st["C"]["rephotos"] += 1
            rephoto = True
            truth = truth_label(a, dirty1, False)
            pred = sample_pred(probs, truth)
            if pred == "unclear_image":             # still unreadable -> expert
                pred = "expert_inspection"
        final_dirty = dirty1
        if pred == "sufficient":
            if dirty1:
                st["C"]["wrongly_clean"] += 1
                st["C"]["dirty_unflagged"] += 1
        elif pred == "more_cleaning":
            st["C"]["revisits"] += 1
            if dirty1 and rng.random() < z["p_clean2"]:
                final_dirty = False
            st["C"]["dirty_unflagged"] += final_dirty
        else:                                       # expert_inspection
            st["C"]["expert_areas"] += 1            # flagged, pending - not silent
        if record_demo:
            demo.append({**a, "fouled": fouled, "dirty_after_pass1": dirty1,
                         "pred_first": pred1, "rephoto": rephoto,
                         "decision": pred, "final_dirty": final_dirty})
    return st, demo


def attach_demo_photos(demo):
    """Give each demo area a before/after pair from the real dataset, matching its decision."""
    rows = list(csv.DictReader(open(HERE / "dataset" / "labels.csv")))
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    idx = {k: 0 for k in by_label}
    for a in demo:
        lab = "unclear_image" if a["rephoto"] and a["decision"] == "expert_inspection" else a["decision"]
        lab = lab if lab in by_label else "sufficient"
        pool = by_label[lab]
        pick = pool[idx[lab] % len(pool)]
        idx[lab] += 1
        a["before_img"] = f"dataset/{pick['before_file']}"
        a["after_img"] = f"dataset/{pick['after_file']}"


def main():
    areas = build_areas()
    cm, probs = load_confusion()

    totals = {k: None for k in "ABC"}
    demo = None
    for i in range(N_HULLS):
        st, d = simulate_hull(areas, probs, record_demo=(i == 0))
        if i == 0:
            demo = d
        for k in "ABC":
            if totals[k] is None:
                totals[k] = dict(st[k])
            else:
                for m in st[k]:
                    totals[k][m] += st[k][m]
    avg = {k: {m: round(v / N_HULLS, 2) for m, v in totals[k].items()} for k in "ABC"}
    attach_demo_photos(demo)

    out = {
        "n_hulls": N_HULLS, "n_areas": len(areas), "grid": {"cols": COLS, "rows": ROWS},
        "zone_assumptions": ZONE_ASSUMPTIONS,
        "measured_confusion_rules_test": cm,
        "strategies": {
            "A": {"name": "A - repeat full route", **avg["A"]},
            "B": {"name": "B - clean once, trust it", **avg["B"]},
            "C": {"name": "C - verify then revisit (ours)", **avg["C"]},
        },
        "demo_hull": demo,
        "notes": [
            "Counts are averages over 1000 simulated hulls; zone probabilities are assumptions.",
            "AI decisions are sampled from the measured step-3 confusion matrix (held-out test).",
            "Context only, not our result: IMO GreenVoyage2050 lists typical full in-water cleaning at $5,000-$50,000.",
        ],
    }
    json.dump(out, open(HERE / "sim_results.json", "w"), indent=1)
    with open(HERE / "sim_results.js", "w") as f:
        f.write("window.SIM_DATA = ")
        json.dump(out, f)
        f.write(";")

    print(f"Simulated {N_HULLS} hulls x {len(areas)} areas. Averages per hull:")
    hdr = ["visits", "checks", "rephotos", "revisits", "expert_areas", "dirty_unflagged", "wrongly_clean"]
    print(f"{'strategy':28}" + "".join(f"{h:>16}" for h in hdr))
    for k in "ABC":
        print(f"{out['strategies'][k]['name']:28}" + "".join(f"{avg[k][h]:>16}" for h in hdr))
    print("Wrote sim_results.json and sim_results.js")


if __name__ == "__main__":
    main()
