"""Plot the mandatory optimization campaign (pto-kernel-optimizer SKILL.md 3.5).

Ships with the plugin so every run produces the SAME graph. Telling an agent to
"plot the trajectory" in prose gets a different chart every run, or none; this makes
it one command with a fixed contract.

    python plot_optimization.py reports/optimization_<stage>.json [-o out.png]
    python plot_optimization.py reports/            # every optimization_*.json in it

Input schema (what Phase 6.5 must write, one file per stage):

    {
      "stage": "attn_scores",
      "archetype": "mixed" | "vec_only" | "cube_only",
      "baseline_ratio": 2.31,            # optional: ratio BEFORE attempt 1
      "attempts": [
        {"n": 1,
         "hypothesis": "L1 double buffering hides the MTE2 latency",
         "changed": "2-slot L1 ping-pong on the K loop",
         "ratio": 1.94,                  # ours/vendor latency, LOWER IS BETTER
         "ci": [1.93, 1.95],             # optional 95% CI
         "kept": true,
         "why": "1.31x, determinism held",
         "kernel": "src/variants/kernel_attn_scores_a01.cpp"},   # optional, see below
        ...
      ],
      "stop_reason": "budget_exhausted" | "hardware_limit",
      "gate": "bandwidth_ceiling",       # required when stop_reason == hardware_limit
      "gate_value": "782 GB/s of a measured 811 GB/s ceiling (96.4%)"
    }

Conventions this script assumes, and enforces in what it draws:
  * ratio = ours / vendor latency, so DOWN IS GOOD and 1.0 is vendor parity;
  * a reverted attempt is still an attempt and is still plotted -- the regressions
    are usually the most informative part of the campaign (full double buffering
    measurably made one kernel SLOWER here, which is the point);
  * the budget is 10, so the x-axis always shows all 10 slots even when fewer were
    used. An early stop is then visible as empty space, which is the intent: you
    should be able to SEE that the campaign stopped short and read why.
"""
import argparse, glob, json, os, sys, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BUDGET = 10
KEPT_C, DROP_C, LINE_C = "#2f5d3a", "#a33", "#4a6fa5"


def plot(doc, out_png):
    att = sorted(doc.get("attempts", []), key=lambda a: a["n"])
    if not att:
        print("  no attempts in %s -- nothing to plot" % out_png, file=sys.stderr)
        return False

    stage = doc.get("stage", "stage")
    arch = doc.get("archetype", "?")
    xs = [a["n"] for a in att]
    ys = [a["ratio"] for a in att]
    kept = [bool(a.get("kept")) for a in att]

    fig, ax = plt.subplots(figsize=(10.2, 5.6))

    # Running best: the line a reader actually cares about, since a campaign keeps
    # the best-so-far kernel and a regression does not undo earlier progress.
    best, run = float("inf"), []
    for y, k in zip(ys, kept):
        if k:
            best = min(best, y)
        run.append(best if best < float("inf") else y)
    ax.step(xs, run, where="post", color=LINE_C, lw=1.6, alpha=.85,
            label="best kept so far", zorder=2)

    base = doc.get("baseline_ratio")
    if base is not None:
        ax.axhline(base, color="#888", ls=":", lw=1.2, zorder=1)
        ax.annotate("baseline %.3f" % base, xy=(0.015, base),
                    xycoords=("axes fraction", "data"), va="bottom", ha="left",
                    fontsize=8.5, color="#666")

    for a, k in zip(att, kept):
        ci = a.get("ci")
        err = [[a["ratio"] - ci[0]], [ci[1] - a["ratio"]]] if ci else None
        ax.errorbar(a["n"], a["ratio"], yerr=err, fmt="o", ms=9, capsize=4,
                    color=KEPT_C if k else DROP_C,
                    mfc=(KEPT_C if k else "none"),
                    mec=(KEPT_C if k else DROP_C), zorder=3)

    ax.axhline(1.0, color="#333", ls="--", lw=1.3, zorder=1)
    ax.annotate("vendor parity", xy=(0.015, 1.0), xycoords=("axes fraction", "data"),
                va="bottom", ha="left", fontsize=9, color="#333")

    # Always show all 10 slots: an early stop should be VISIBLE as unused budget.
    ax.set_xlim(0.4, BUDGET + 0.6)
    ax.set_xticks(range(1, BUDGET + 1))
    used = len(att)
    if used < BUDGET:
        ax.axvspan(used + 0.5, BUDGET + 0.6, color="#bbb", alpha=.14, zorder=0)
        ax.annotate("budget not used (%d of %d)" % (used, BUDGET),
                    xy=((used + 0.5 + BUDGET + 0.6) / 2, 0.965),
                    xycoords=("data", "axes fraction"), ha="center",
                    fontsize=8.5, color="#666")

    ax.set_xlabel("optimization attempt  (budget %d)" % BUDGET)
    ax.set_ylabel("latency ratio  ours / vendor   (lower is better)")
    ax.set_title("%s -- optimization campaign  [%s]" % (stage, arch),
                 fontsize=12.5, pad=30, loc="left")

    stop = doc.get("stop_reason", "unspecified")
    sub = "stop: %s" % stop
    if stop == "hardware_limit":
        sub += "  --  gate %s: %s" % (doc.get("gate", "?"), doc.get("gate_value", "?"))
    elif used < BUDGET and arch == "mixed":
        sub += "  --  PROCESS FAILURE: a mixed stage must run all %d attempts" % BUDGET
    ax.text(0.0, 1.012, "\n".join(textwrap.wrap(sub, 110)), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.5,
            color=("#a33" if "PROCESS FAILURE" in sub else "#555"), linespacing=1.4)

    ax.grid(axis="y", alpha=.25)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=KEPT_C, mec=KEPT_C, label="kept"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=DROP_C,
               label="reverted / regressed"),
        Line2D([], [], color=LINE_C, lw=1.6, label="best kept so far"),
    ], fontsize=9, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("  wrote %s  (%d/%d attempts, stop: %s)" % (out_png, used, BUDGET, stop))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="optimization_<stage>.json, or a reports/ directory")
    ap.add_argument("-o", "--out", help="output PNG (single-file mode only)")
    a = ap.parse_args()

    files = ([a.path] if a.path.endswith(".json")
             else sorted(glob.glob(os.path.join(a.path, "optimization_*.json"))))
    if not files:
        print("no optimization_*.json found under %s" % a.path, file=sys.stderr)
        return 1

    ok = True
    for f in files:
        doc = json.load(open(f))
        out = a.out if (a.out and len(files) == 1) else \
            f[:-len(".json")].replace("optimization_", "optimization_trajectory_") + ".png"
        ok = plot(doc, out) and ok
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
