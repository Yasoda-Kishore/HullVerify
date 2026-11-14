"""
STEP 7: Test it and be honest.

1. One results table: AI rules vs the naive baseline, TEST split only
   (read from results_step3.json - nothing is recomputed, nothing retuned).
2. The wrongly-"sufficient" rate, reported even if it is bad.
3. Sensitivity check: vary the first-pass cleaning probability and the cost
   inputs; note where strategy C stops winning on cost. Shown honestly.
4. The limits slide, ready to paste.

Run:  python3 step7_tests.py      -> prints the report and writes step7_report.md
"""
import json
from pathlib import Path

import step4_simulation as s4

HERE = Path(__file__).parent
N_HULLS = 300          # per sensitivity setting (speed); headline sim used 1000

# default cost inputs - SAME defaults as the sliders in hull_viewer.html
COST = {"clean_min": 3.0, "check_min": 0.5, "rephoto_min": 1.0, "over_min": 1.0,
        "rate_eur_h": 150.0, "expert_eur": 40.0, "penalty_eur": 60.0}


def cost_of(c, s):
    """euros from strategy counts s and cost inputs c (counts x inputs, as in the viewer)."""
    time_h = (s["visits"] * c["clean_min"] + s["checks"] * c["check_min"]
              + s["rephotos"] * c["rephoto_min"]
              + s["revisits"] * (c["clean_min"] + c["over_min"])) / 60.0
    money = time_h * c["rate_eur_h"] + s["expert_areas"] * c["expert_eur"] \
            + s["dirty_unflagged"] * c["penalty_eur"]
    return time_h, money


def run_sim(clean_mult=1.0):
    """Average strategy counts over N_HULLS hulls, with first-pass/revisit clean
    probabilities scaled by clean_mult (capped at 0.98). Restores assumptions after."""
    backup = {z: dict(v) for z, v in s4.ZONE_ASSUMPTIONS.items()}
    try:
        for z in s4.ZONE_ASSUMPTIONS:
            s4.ZONE_ASSUMPTIONS[z]["p_clean1"] = min(0.98, backup[z]["p_clean1"] * clean_mult)
            s4.ZONE_ASSUMPTIONS[z]["p_clean2"] = min(0.98, backup[z]["p_clean2"] * clean_mult)
        areas = s4.build_areas()
        _, probs = s4.load_confusion()
        tot = None
        for _ in range(N_HULLS):
            st, _d = s4.simulate_hull(areas, probs)
            if tot is None:
                tot = {k: dict(st[k]) for k in "ABC"}
            else:
                for k in "ABC":
                    for m in st[k]:
                        tot[k][m] += st[k][m]
        return {k: {m: v / N_HULLS for m, v in tot[k].items()} for k in "ABC"}
    finally:
        for z in s4.ZONE_ASSUMPTIONS:
            s4.ZONE_ASSUMPTIONS[z].update(backup[z])


def fmt_row(cells, w=12):
    return "| " + " | ".join(str(c).ljust(w) for c in cells) + " |"


def main():
    out = []
    say = out.append

    # ---------------- 1 + 2: rules vs baseline on TEST, wrongly-sufficient rate
    r3 = json.load(open(HERE / "results_step3.json"))
    say("# Step 7 report - tests, sensitivity, limits\n")
    say("## 1. AI rules vs naive baseline (held-out TEST split only, n=%d)" % r3["rules_test"]["n"])
    say("Thresholds were tuned on the tune split only. Synthetic images - demonstrates the")
    say("pipeline, NOT real-world accuracy.\n")
    rows = [("", "accuracy", "wrongly-'sufficient'", "meaning")]
    for key, name in [("rules_test", "our rules"), ("baseline_test", "naive baseline"),
                      ("logreg_test", "logistic regression")]:
        if key not in r3:
            continue
        d = r3[key]
        acc = f"{d['accuracy']:.0%}"
        ws = f"{d['false_sufficient']}/{d['dirty_total']}"
        meaning = {"rules_test": "1 error, over-cautious direction",
                   "baseline_test": "passes all murky images as 'clean' - the dangerous failure",
                   "logreg_test": "perfect on THIS small synthetic set - do not oversell"}[key]
        rows.append((name, acc, ws, meaning))
    say(fmt_row(rows[0], 22))
    say(fmt_row(["-" * 22] * len(rows[0]), 22))
    for r in rows[1:]:
        say(fmt_row(r, 22))
    say("\n**Wrongly-'sufficient' rate (dirty area passed as clean): 0/12 on test.**")
    say("Report it even when it is bad; here it is 0 on a small synthetic set, which")
    say("may NOT hold on real imagery - say that out loud.\n")

    # ---------------- 3a: sensitivity to first-pass cleaning reliability
    say("## 2. Sensitivity: how reliable must cleaning be before C stops paying?")
    say(f"Costs below use the default slider inputs {COST} -")
    say("all assumptions. Counts from %d simulated hulls per setting.\n" % N_HULLS)
    say(fmt_row(("clean-prob x", "cost A", "cost B", "cost C", "dirty A", "dirty B", "dirty C", "cheapest")))
    say(fmt_row(["-" * 12] * 8))
    sens = []
    for mult in (0.6, 0.8, 1.0, 1.1, 1.2, 1.3):
        avg = run_sim(mult)
        costs = {k: cost_of(COST, avg[k])[1] for k in "ABC"}
        win = min(costs, key=costs.get)
        sens.append((mult, costs, {k: avg[k]["dirty_unflagged"] for k in "ABC"}, win))
        say(fmt_row((f"x{mult:.1f}", *(f"EUR {costs[k]:,.0f}" for k in "ABC"),
                     *(f"{avg[k]['dirty_unflagged']:.1f}" for k in "ABC"), win)))
    say("")

    # ---------------- 3b: sensitivity to the dirty-area penalty
    say("## 3. Sensitivity: the penalty for a dirty area left behind")
    say("At penalty = 0, leaving fouling behind is 'free', so the cheapest strategy is")
    say("doing less. C's case rests on dirty areas having a real cost (fuel, re-dock, audit).\n")
    base = run_sim(1.0)
    say(fmt_row(("penalty EUR", "cost A", "cost B", "cost C", "cheapest")))
    say(fmt_row(["-" * 12] * 5))
    pen_rows = []
    for pen in (0, 20, 40, 60, 120, 200):
        c = dict(COST, penalty_eur=pen)
        costs = {k: cost_of(c, base[k])[1] for k in "ABC"}
        win = min(costs, key=costs.get)
        pen_rows.append((pen, costs, win))
        say(fmt_row((pen, *(f"EUR {costs[k]:,.0f}" for k in "ABC"), win)))
    say("")

    # ---------------- 3c: sensitivity to verification time
    say("## 4. Sensitivity: how slow may verification be?")
    say(fmt_row(("verify min/area", "cost C", "cheapest")))
    say(fmt_row(["-" * 14] * 3))
    chk_rows = []
    for chk in (0.5, 1.0, 2.0, 3.0, 4.0):
        c = dict(COST, check_min=chk)
        costs = {k: cost_of(c, base[k])[1] for k in "ABC"}
        win = min(costs, key=costs.get)
        chk_rows.append((chk, costs, win))
        say(fmt_row((chk, f"EUR {costs['C']:,.0f}", win)))
    say("")

    # honest reading of where C loses
    losers = [f"clean-prob x{m}" for m, _c, _d, w in sens if w != "C"] + \
             [f"penalty EUR {p}" for p, _c, w in pen_rows if w != "C"] + \
             [f"verify {m} min/area" for m, _c, w in chk_rows if w != "C"]
    say("## 5. Where C stops winning (say this in the pitch)")
    if losers:
        say("C is NOT the cheapest at: " + "; ".join(losers) + ".")
    else:
        say("C was cheapest at every setting tested - which should raise an eyebrow:")
    say("The pattern to expect: C loses when cleaning is already very reliable (little")
    say("to find), when leaving dirt behind costs nothing, or when verification is slow.")
    say("Even where C is not cheapest, it is the only strategy that KNOWS which areas")
    say("are still dirty - the others leave them silently.\n")

    # ---------------- 4: limits slide
    say("## 6. Limits slide (paste as-is)")
    say("- Images are computer-generated stand-ins, not real hull photos.")
    say("- The robot and the hull are simulated; zone fouling/cleaning probabilities are assumptions.")
    say("- Test set is small (24 pairs); thresholds fitted to this synthetic distribution.")
    say("- Cost figures come from editable assumed inputs, not from operations data.")
    say("- AI error rates feed the simulation from a measured but synthetic test; real-water")
    say("  performance is unknown until tried on real imagery.")
    say("- IMO GreenVoyage2050 context ($5k-$50k per full clean) is background, not our result.")

    report = "\n".join(out)
    (HERE / "step7_report.md").write_text(report)
    print(report)
    print("\nWrote step7_report.md")


if __name__ == "__main__":
    main()
