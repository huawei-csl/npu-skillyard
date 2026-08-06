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
         "correct": true,             # OPTIONAL, default true. FALSE = failed validation.
         "kind": "candidate",         # OPTIONAL, default "candidate". Or "diagnostic".
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
  * a kernel that FAILED VALIDATION must never read as a win. A real run produced an
    attempt that measured as the fastest point on the whole chart and was numerically
    wrong (0/14 cases, 2.65M elements off by more than 1). The eye goes straight to
    the lowest point, so those are drawn as red crosses, excluded from the best-kept
    line, and called out in a banner. Speed for a wrong kernel is not a result;
  * a DIAGNOSTIC (a noop-floor probe, a strided-vs-contiguous test) is a measurement,
    not a shippable kernel. Plotting it as a candidate conflates "something I
    measured" with "something I could have shipped", so it is drawn separately;
  * the budget is 15, so the x-axis always shows all 15 slots even when fewer were
    used. An early stop is then visible as empty space, which is the intent: you
    should be able to SEE that the campaign stopped short and read why.
"""
import argparse, glob, json, os, sys, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BUDGET_DEFAULT = 15


def _budget(doc, att):
    """Budget comes from the JSON, never from a constant in this file.

    A hard-coded budget silently CLIPS a campaign that ran longer: one run made 24
    attempts against a nominal 15 and lost attempts 11-24 off the chart. And the
    constant goes stale every time the mandate changes. Take the declared budget if
    the campaign wrote one, and always stretch to cover the attempts actually made,
    so nothing measured is ever dropped.
    """
    declared = doc.get("budget") or doc.get("attempt_budget") or BUDGET_DEFAULT
    highest = max((a.get("n") or 0) for a in att) if att else 0
    return max(int(declared), int(highest))
KEPT_C, DROP_C, LINE_C = "#2f5d3a", "#a33", "#4a6fa5"
BAD_C, DIAG_C = "#c1121f", "#888"


def _scalarize(v, what, n):
    """Accept a per-sweep-point {size: ratio} mapping as well as a scalar.

    A campaign that benchmarks several sizes naturally records `ratio` (and `ci`)
    per size. Crashing on that shape cost a pipeline run its trajectory plot, so
    collapse to the PRODUCTION point -- the largest numeric key, which is the size
    the contract's coverage gate cares about -- and say so rather than silently
    picking one.
    """
    if v is None or isinstance(v, (int, float)):
        return v
    if isinstance(v, dict):
        if not v:
            return None
        try:
            key = max(v, key=lambda k: float(k))
        except (TypeError, ValueError):
            key = sorted(v)[-1]
        print("  attempt %s: %s is per-size; plotting the production point (%s)"
              % (n, what, key), file=sys.stderr)
        return _scalarize(v[key], what, n)
    if isinstance(v, (list, tuple)) and what == "ratio":
        return _scalarize(v[-1], what, n) if v else None
    return v


def plot(doc, out_png):
    att = sorted(doc.get("attempts", []), key=lambda a: a["n"])
    # Normalise BEFORE any arithmetic: `ratio` and `ci` may arrive per sweep point.
    for a in att:
        a["ratio"] = _scalarize(a.get("ratio"), "ratio", a.get("n"))
        ci = a.get("ci")
        if isinstance(ci, dict):
            ci = _scalarize(ci, "ci", a.get("n"))
        if isinstance(ci, (list, tuple)) and len(ci) == 2 and \
                all(isinstance(x, (int, float)) for x in ci):
            a["ci"] = list(ci)
        elif ci is not None:
            a["ci"] = None          # unusable shape -> draw the point without a bar
    if isinstance(doc.get("baseline_ratio"), dict):
        doc["baseline_ratio"] = _scalarize(doc["baseline_ratio"], "baseline_ratio", 0)
    # An attempt with no ratio is legitimate and must NOT crash the plot: a change
    # that failed to COMPILE, or whose validation aborted, never produced a number.
    # It is still an attempt and still consumed budget, so it is kept in the count
    # and drawn as a marker on the "no measurement" rule at the top of the axis.
    unmeasured = [a for a in att if a.get("ratio") is None]
    att = [a for a in att if a.get("ratio") is not None]
    if not att:
        print("  %s: %d attempt(s), none with a ratio -- nothing to plot"
              % (out_png, len(unmeasured)), file=sys.stderr)
        return False

    budget = _budget(doc, att + unmeasured)
    stage = doc.get("stage", "stage")
    arch = doc.get("archetype", "?")
    xs = [a["n"] for a in att]
    ys = [a["ratio"] for a in att]
    kept = [bool(a.get("kept")) for a in att]
    # Default to correct/candidate so older JSONs keep plotting unchanged.
    ok = [a.get("correct", True) is not False for a in att]
    diag = [a.get("kind", "candidate") == "diagnostic" for a in att]

    fig, ax = plt.subplots(figsize=(10.2, 5.6))

    base = doc.get("baseline_ratio")

    # Running best: the line a reader actually cares about, since a campaign keeps
    # the best-so-far kernel and a regression does not undo earlier progress.
    #
    # SEED IT WITH THE BASELINE. A campaign where nothing is kept is a real and
    # informative outcome (one case here went 10/10 with 0 kept), and seeding with
    # +inf made the line fall back to tracing the raw ratios -- so it wandered up
    # and down while the legend called it "best kept so far". Held at the baseline
    # it says the true thing: the shipped kernel never moved.
    best, run = (base if base is not None else float("inf")), []
    for y, k, c, dg in zip(ys, kept, ok, diag):
        if k and c and not dg:          # only a CORRECT, KEPT candidate moves the best
            best = min(best, y)
        run.append(best if best < float("inf") else y)
    ax.step(xs, run, where="post", color=LINE_C, lw=1.6, alpha=.85,
            label="best kept so far", zorder=2)
    # Warnings go ABOVE the axes, never over the data. Drawing them at a fixed
    # axes fraction put them straight through the points whenever the campaign
    # ended near the bottom of the range -- which is exactly when it went well.
    warn = []
    if not any(kept):
        warn.append("no attempt was kept -- shipped kernel is the baseline")
    valid_ys = [y for y, c, dg in zip(ys, ok, diag) if c and not dg]
    bad_ys = [y for y, c in zip(ys, ok) if not c]
    if bad_ys and (not valid_ys or min(bad_ys) < min(valid_ys)):
        warn.append("an attempt at or below the best valid result FAILED VALIDATION "
                    "-- it is not a result")

    if base is not None:
        ax.axhline(base, color="#888", ls=":", lw=1.2, zorder=1)
        ax.annotate("baseline %.3f" % base, xy=(0.015, base),
                    xycoords=("axes fraction", "data"), va="bottom", ha="left",
                    fontsize=8.5, color="#666")

    for a, k, c, dg in zip(att, kept, ok, diag):
        ci = a.get("ci")
        err = [[a["ratio"] - ci[0]], [ci[1] - a["ratio"]]] if ci else None
        if not c:            # failed validation -- must not read as a win
            style = dict(fmt="X", ms=11, color=BAD_C, mec=BAD_C, mfc=BAD_C)
        elif dg:             # a measurement, not a shippable kernel
            style = dict(fmt="s", ms=8, color=DIAG_C, mec=DIAG_C, mfc="none")
        else:
            style = dict(fmt="o", ms=9, color=KEPT_C if k else DROP_C,
                         mfc=(KEPT_C if k else "none"),
                         mec=(KEPT_C if k else DROP_C))
        ax.errorbar(a["n"], a["ratio"], yerr=err, capsize=4, zorder=3, **style)

    ax.axhline(1.0, color="#333", ls="--", lw=1.3, zorder=1)
    ax.annotate("vendor parity", xy=(0.015, 1.0), xycoords=("axes fraction", "data"),
                va="bottom", ha="left", fontsize=9, color="#333")

    # Always show all 15 slots: an early stop should be VISIBLE as unused budget.
    ax.set_xlim(0.4, budget + 0.6)
    ax.set_xticks(range(1, budget + 1))
    # SKILL 3.5 defines an attempt as "a change WITH a paired re-measurement", and a
    # diagnostic probe is exactly that -- the skill tells you to TAG probes
    # `kind: diagnostic`, not to exclude them from the budget. Excluding them made a
    # full 15-attempt campaign (3 of them tagged probes) render as "12 of 15" under a
    # red PROCESS FAILURE banner. Budget consumption is the number of attempt SLOTS
    # occupied, so take the highest attempt index present rather than a count -- that
    # also keeps the shading right when indices are sparse.
    used = max([a["n"] for a in att] + [a["n"] for a in unmeasured] + [0])
    if used < budget:
        ax.axvspan(used + 0.5, budget + 0.6, color="#bbb", alpha=.14, zorder=0)
        ax.annotate("budget not used (%d of %d)" % (used, budget),
                    xy=((used + 0.5 + budget + 0.6) / 2, 0.965),
                    xycoords=("data", "axes fraction"), ha="center",
                    fontsize=8.5, color="#666")

    if unmeasured:
        top = max(ys) + 0.06 * (max(ys) - min(ys) + 1e-9)
        for a in unmeasured:
            ax.plot(a["n"], top, marker="$?$", ms=11, color="#777", zorder=3)
        warn.append("? = attempt produced no measurement (build or validation aborted)")
    ax.set_xlabel("optimization attempt  (budget %d)" % budget)
    ax.set_ylabel("latency ratio  ours / vendor   (lower is better)")
    stop = doc.get("stop_reason", "unspecified")
    sub = "stop: %s" % stop
    if stop == "hardware_limit":
        sub += "  --  gate %s: %s" % (doc.get("gate", "?"), doc.get("gate_value", "?"))
    elif used < budget and arch == "mixed":
        warn.append("PROCESS FAILURE: a mixed stage must run all %d attempts" % budget)
    lines = textwrap.wrap(sub, 110)

    # Title pad must clear the banner: the stop/gate text wraps to an arbitrary
    # number of lines and the warning column sits at the same baseline, so a
    # fixed pad lets a long gate_value collide with the title.
    banner_lines = max(len(lines), len(warn), 1)
    ax.set_title("%s -- optimization campaign  [%s]" % (stage, arch),
                 fontsize=12.5, pad=16 + 11.5 * banner_lines, loc="left")
    ax.text(0.0, 1.012, "\n".join(lines), transform=ax.transAxes, ha="left",
            va="bottom", fontsize=8.5, color="#555", linespacing=1.4)
    if warn:
        ax.text(1.0, 1.012, "\n".join(warn), transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8.5, color=BAD_C, weight="bold",
                linespacing=1.4)

    ax.grid(axis="y", alpha=.25)
    handles = [
        Line2D([], [], marker="o", ls="", mfc=KEPT_C, mec=KEPT_C, label="kept"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=DROP_C,
               label="reverted / regressed"),
        Line2D([], [], color=LINE_C, lw=1.6, label="best kept so far"),
    ]
    if not all(ok):
        handles.append(Line2D([], [], marker="X", ls="", color=BAD_C,
                              label="FAILED VALIDATION (speed is meaningless)"))
    if any(diag):
        handles.append(Line2D([], [], marker="s", ls="", mfc="none", mec=DIAG_C,
                              label="diagnostic probe, not a candidate"))
    ax.legend(handles=handles, fontsize=9, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("  wrote %s  (%d/%d attempts, stop: %s)" % (out_png, used, budget, stop))
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
