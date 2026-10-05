"""
Step 6: did students rate AI-enhanced matches higher than baseline matches?

    python analyze.py            # real data: data/ratings.csv + data/assignments.json
    python analyze.py --sample   # SIMULATED ratings (see simulate_ratings.py)

If you collected ratings in a Google Sheet, download it as CSV
(File -> Download -> Comma-separated values) and save it as data/ratings.csv.

Outputs: a printed report, results/summary.json and results/results_chart.png
(results/ holds only aggregate numbers, no personal data).
"""
import argparse
import csv
import itertools
import json
import math
import os
import random
import statistics

import config

RESULTS_DIR = "results"
AI_COLOR, BASELINE_COLOR = "#D9692F", "#3A78B5"   # checked for colour-blind separation


# ------------------------------------------------------------ loading
def load_ratings(path):
    """Keep each student's LATEST answer per match (they may submit twice)."""
    latest = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row["rater_id"], row["match_id"])
            if key not in latest or row["timestamp"] > latest[key]["timestamp"]:   # ISO times sort as text
                latest[key] = row
    return list(latest.values())


def join_with_assignments(ratings, assignments):
    """Attach the hidden label (ai / baseline) to every rating."""
    labels = {(rater, a["match_id"]): a for rater, items in assignments["assignments"].items() for a in items}
    joined, unmatched = [], 0
    for row in ratings:
        label = labels.get((row["rater_id"], row["match_id"]))
        if label is None:
            unmatched += 1      # e.g. a test submission; skip it
            continue
        joined.append({"rater": row["rater_id"], "source": label["source"],
                       "rank": label["rank_in_source"], "rating": int(row["rating"]),
                       "message": row["would_message"].strip().lower() == "yes"})
    return joined, unmatched


# ------------------------------------------------------------ statistics
def sign_flip_test(diffs, n_sims=100_000, seed=0):
    """Paired permutation test (two-sided).

    If AI and baseline were equally good, each student's difference would be
    just as likely to be positive as negative. So we flip the signs in every
    possible way (or 100,000 random ways when n is large) and ask: how often
    is the average difference at least as far from 0 as the one we observed?
    That fraction is the p-value. No assumption that ratings are normal."""
    n = len(diffs)
    observed = abs(sum(diffs)) / n
    if n <= 16:                                           # 2^16 = 65,536 patterns: do them all
        patterns, total = itertools.product((1, -1), repeat=n), 2 ** n
    else:
        rng = random.Random(seed)
        patterns = ([rng.choice((1, -1)) for _ in range(n)] for _ in range(n_sims))
        total = n_sims
    extreme = sum(1 for signs in patterns
                  if abs(sum(s * d for s, d in zip(signs, diffs))) / n >= observed - 1e-12)
    return extreme / total


def bootstrap_ci(diffs, n_boot=10_000, seed=0):
    """95% interval for the mean difference: resample students with replacement
    10,000 times and take the middle 95% of the resampled means."""
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def mean_ci(values):
    """Mean with a rough 95% interval (normal approximation), for the chart."""
    if len(values) < 2:
        return statistics.mean(values), 0.0
    return statistics.mean(values), 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def pct(part, whole):
    return 100 * part / whole if whole else float("nan")


# ------------------------------------------------------------ the analysis
def analyze(joined):
    by_source = {s: [r for r in joined if r["source"] == s] for s in ("ai", "baseline")}

    # Each student's AI pick that ranked highest in the AI list ("top AI match").
    best_ai = {}
    for r in by_source["ai"]:
        if r["rater"] not in best_ai or r["rank"] < best_ai[r["rater"]]["rank"]:
            best_ai[r["rater"]] = r
    baseline_of = {r["rater"]: r for r in by_source["baseline"]}

    # 1. top AI match rated 4 or 5
    top_ai = list(best_ai.values())
    top_ai_good = sum(r["rating"] >= 4 for r in top_ai)

    # 2 + 3. averages and "would message", over all ratings
    summary = {}
    for source, rows in by_source.items():
        summary[source] = {
            "ratings": len(rows),
            "mean_rating": statistics.mean(r["rating"] for r in rows) if rows else float("nan"),
            "pct_would_message": pct(sum(r["message"] for r in rows), len(rows)),
        }

    # 4. paired, like for like: each student's best AI pick vs their baseline pick
    #    (each algorithm's best match that the other algorithm didn't rank top 3).
    pairs = {s: (best_ai[s]["rating"], baseline_of[s]["rating"]) for s in best_ai if s in baseline_of}
    diffs = [ai - base for ai, base in pairs.values()]
    raters = {r["rater"] for r in joined}

    # Secondary: the average of BOTH AI picks vs baseline (uses all the data,
    # but AI's second pick is usually weaker, which tilts it towards baseline).
    both = {}
    for r in joined:
        both.setdefault(r["rater"], {"ai": [], "baseline": []})[r["source"]].append(r["rating"])
    both_diffs = [statistics.mean(v["ai"]) - statistics.mean(v["baseline"])
                  for v in both.values() if v["ai"] and v["baseline"]]

    paired = {"n_students": len(diffs)}
    if len(diffs) >= 2:
        sd = statistics.stdev(diffs)
        low, high = bootstrap_ci(diffs)
        paired.update({
            "mean_difference": statistics.mean(diffs),
            "ci95": [low, high],
            "p_value": sign_flip_test(diffs),
            "ai_higher": sum(d > 0 for d in diffs),
            "tied": sum(d == 0 for d in diffs),
            "baseline_higher": sum(d < 0 for d in diffs),
            # Smallest true difference we'd detect 80% of the time at p < 0.05 with this n and spread
            # (normal approximation: (1.96 + 0.84) x sd / sqrt(n)).
            "min_detectable_difference": 2.8 * sd / math.sqrt(len(diffs)) if sd > 0 else 0.0,
        })

    if len(both_diffs) >= 2:
        paired["secondary_both_ai_picks"] = {"mean_difference": statistics.mean(both_diffs),
                                             "p_value": sign_flip_test(both_diffs)}

    return {
        "raters": len(raters),
        "top_ai_match": {"rated": len(top_ai), "rated_4_or_5": top_ai_good,
                         "pct_4_or_5": pct(top_ai_good, len(top_ai))},
        "by_source": summary,
        "paired": paired,
    }, pairs


def interpretation(result, design_counts):
    """Plain-language guard rails, so the numbers aren't over-claimed."""
    p, n = result["paired"], result["paired"]["n_students"]
    lines = []
    if n < 10:
        lines.append(f"Only {n} students rated both kinds of match: too few to conclude anything. Treat this as a pilot.")
    elif "p_value" in p:
        low, high = p["ci95"]
        if p["p_value"] < 0.05:
            better = "AI-enhanced" if p["mean_difference"] > 0 else "baseline"
            lines.append(f"The {better} matches were rated higher, and a difference this big would be "
                         f"unlikely (p = {p['p_value']:.3f}) if the two methods were equally good.")
        else:
            lines.append(f"No reliable difference (p = {p['p_value']:.2f}). This does NOT show the methods are "
                         f"equal: with {n} students, only a difference of about "
                         f"{p['min_detectable_difference']:.1f} points or more would reliably show up.")
        lines.append(f"The true average difference is plausibly anywhere from {low:+.2f} to {high:+.2f} points.")
    lines.append("This compares matches where the two methods DISAGREED (each side's best pick that the other "
                 "didn't rank top 3). It says which method wins disagreements, not how good either is overall.")
    fallback = design_counts.get("fallback", 0)
    if fallback:
        lines.append(f"For {fallback} student(s) the two lists barely differed, so they got AI's top 2 plus "
                     f"baseline's best remaining pick instead (slightly favours AI).")
    lines.append("Ratings are first impressions of a profile, not proof people would live well together.")
    return lines


# ------------------------------------------------------------ the chart
def make_chart(result, pairs, path, simulated):
    import matplotlib
    matplotlib.use("Agg")                      # draw to a file, no window needed
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1F2328", "#6B7280", "#E5E7EB"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": ink,
                         "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("white")

    # Left: each student's baseline vs AI rating (grey lines), plus the averages.
    rng = random.Random(1)
    for ai, base in pairs.values():
        jitter = rng.uniform(-0.06, 0.06)     # tiny vertical nudge so overlapping lines stay visible
        left.plot([0, 1], [base + jitter, ai + jitter], color="#C9CDD3", linewidth=1, zorder=1)
    for x, source, color in [(0, "baseline", BASELINE_COLOR), (1, "ai", AI_COLOR)]:
        values = [p[1] if source == "baseline" else p[0] for p in pairs.values()]
        mean, err = mean_ci(values)
        left.errorbar(x, mean, yerr=err, fmt="o", color=color, markersize=10, capsize=0,
                      elinewidth=2, zorder=3, markeredgecolor="white", markeredgewidth=2)
        left.annotate(f"{mean:.2f}", (x, mean), xytext=(14 if x else -14, 0), textcoords="offset points",
                      ha="left" if x else "right", va="center", fontsize=12, fontweight="bold", color=ink)
    left.set_xticks([0, 1], ["Baseline's best pick", "AI's best pick"])
    left.set_xlim(-0.5, 1.5)
    left.set_ylim(0.6, 5.4)
    left.set_yticks([1, 2, 3, 4, 5])
    left.set_ylabel("Rating (1 = poor, 5 = great)")
    p = result["paired"]
    left.set_title(f"Match rating (each line = one of {p['n_students']} students)",
                   loc="left", fontsize=12, color=ink, pad=12)

    # Right: % who'd message the match.
    sources = [("baseline", "Baseline", BASELINE_COLOR), ("ai", "AI-enhanced", AI_COLOR)]
    values = [result["by_source"][s]["pct_would_message"] for s, _, _ in sources]
    bars = right.bar([label for _, label, _ in sources], values, color=[c for _, _, c in sources], width=0.55)
    for bar, value, (s, _, _) in zip(bars, values, sources):
        n = result["by_source"][s]["ratings"]
        right.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%", ha="center",
                   fontsize=12, fontweight="bold", color=ink)
        right.text(bar.get_x() + bar.get_width() / 2, 3, f"{n} ratings", ha="center", fontsize=9, color="white")
    right.set_ylim(0, 105)
    right.set_ylabel("% of matches")
    right.set_title("Would want to message them", loc="left", fontsize=12, color=ink, pad=12)

    for ax in (left, right):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.yaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    title = "MeetKat blind test: AI-enhanced vs baseline flatmate matches"
    if simulated:
        title = "SIMULATED DATA · " + title
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=14, fontweight="bold", color=ink)
    footer = ("Each student saw 3 unlabeled matches in random order: 2 from AI-enhanced matching, 1 from baseline,"
              "\neach one a match the other method didn't rank in its top 3.")
    if "p_value" in p:
        low, high = p["ci95"]
        footer += (f"\nPaired difference {p['mean_difference']:+.2f} points "
                   f"(95% CI {low:+.2f} to {high:+.2f}), permutation test p = {p['p_value']:.2f}.")
    fig.text(0.01, 0.015, footer, fontsize=9, color=muted, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.99))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description="Compare AI vs baseline ratings.")
    parser.add_argument("--sample", action="store_true", help="use SIMULATED ratings in sample_data/")
    args = parser.parse_args()
    paths = config.paths(args.sample)

    with open(paths["assignments"], encoding="utf-8") as f:
        assignments = json.load(f)
    joined, unmatched = join_with_assignments(load_ratings(paths["ratings"]), assignments)
    result, pairs = analyze(joined)
    rank_used = assignments["meta"].get("rank_used", {})
    notes = interpretation(result, assignments["meta"].get("design_counts", {}))

    s, p, top = result["by_source"], result["paired"], result["top_ai_match"]
    print("=" * 60)
    print("MeetKat results" + (" (SIMULATED)" if args.sample else ""))
    print("=" * 60)
    print(f"Students who rated: {result['raters']}   (ignored {unmatched} unmatched rows)")
    print(f"Top AI match rated 4 or 5: {top['rated_4_or_5']} of {top['rated']} ({top['pct_4_or_5']:.0f}%)")
    for source in ("ai", "baseline"):
        print(f"{source:>9}: average {s[source]['mean_rating']:.2f} / 5  ·  would message "
              f"{s[source]['pct_would_message']:.0f}%  ({s[source]['ratings']} ratings)")
    if "p_value" in p:
        print(f"Paired (n = {p['n_students']} students): AI - baseline = {p['mean_difference']:+.2f} points, "
              f"95% CI [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}], p = {p['p_value']:.3f}")
        print(f"  AI higher for {p['ai_higher']}, tied for {p['tied']}, baseline higher for {p['baseline_higher']}")
    if "secondary_both_ai_picks" in p:
        sec = p["secondary_both_ai_picks"]
        print(f"Secondary (average of both AI picks vs baseline): {sec['mean_difference']:+.2f} points, "
              f"p = {sec['p_value']:.3f}")
    print(f"Rank of shown picks in their own list: {rank_used}")
    print()
    for line in notes:
        print("- " + line)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    prefix = "SIMULATED_" if args.sample else ""
    with open(os.path.join(RESULTS_DIR, prefix + "summary.json"), "w", encoding="utf-8") as f:
        json.dump({**result, "rank_used": rank_used, "notes": notes}, f, indent=2)
    if len(pairs) >= 2:
        chart = os.path.join(RESULTS_DIR, prefix + "results_chart.png")
        make_chart(result, pairs, chart, args.sample)
        print(f"\nChart saved to {chart}")


if __name__ == "__main__":
    main()
