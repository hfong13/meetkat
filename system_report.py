"""
Measure how the matching system behaves on a set of responses.

    python system_report.py --sample     # the 30 synthetic Hong Kong students
    python system_report.py              # real data

Reports how each hard filter narrows the pool, how many matches each student
gets, how often the AI and baseline modes disagree, and whether the planted
test pairs (sample data only) were handled correctly. These are measurements
of the SYSTEM, not of user satisfaction.
"""
import argparse
import itertools
import json
import os
import statistics

import config
import matcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    paths = config.paths(args.sample)
    with open(paths["profiles"], encoding="utf-8") as f:
        data = json.load(f)
    profiles = data["profiles"]
    n = len(profiles)
    report = {"students": n, "trait_source": data["meta"]["model"]}

    # 1. Filter funnel: apply the hard filters one after another to every pair.
    pairs = list(itertools.combinations(profiles, 2))
    remaining = pairs
    funnel = [("All possible pairs", len(pairs))]
    for label, ok in [
        ("Budgets overlap", lambda a, b: matcher.budgets_overlap(a, b)),
        ("Move-in within 6 weeks", lambda a, b: matcher.move_in_compatible(a, b)),
        ("No dealbreaker either way", lambda a, b: not matcher.violates_dealbreakers(a, b)
                                                   and not matcher.violates_dealbreakers(b, a)),
    ]:
        remaining = [(a, b) for a, b in remaining if ok(a["structured"], b["structured"])]
        funnel.append((label, len(remaining)))
    report["funnel"] = funnel

    # 2. Coverage and scores
    graphs = {mode: matcher.build_graph(profiles, mode) for mode in ("ai", "baseline")}
    degrees = [len(v) for v in graphs["ai"].values()]
    report["eligible_matches_per_student"] = {
        "mean": statistics.mean(degrees), "min": min(degrees), "max": max(degrees),
        "students_with_3_or_more": sum(d >= 3 for d in degrees)}
    edges = [e for v in graphs["ai"].values() for e in v.values()]
    gaps = [abs(e["my_view"] - e["their_view"]) for e in edges]
    report["one_sided_pairs"] = {
        "share_with_20pt_gap": sum(g >= 20 for g in gaps) / len(gaps),
        "avg_points_removed_by_harmonic_mean":
            statistics.mean((e["my_view"] + e["their_view"]) / 2 - e["score"] for e in edges)}

    # 3. Do the two modes disagree?
    order = {p["user_id"]: i for i, p in enumerate(profiles)}
    overlaps, top1_differs = [], 0
    for uid in graphs["ai"]:
        ai = matcher.top_k(graphs["ai"][uid], 3, order)
        base = matcher.top_k(graphs["baseline"][uid], 3, order)
        if ai and base:
            overlaps.append(len(set(ai) & set(base)))
            top1_differs += ai[0] != base[0]
    report["ai_vs_baseline"] = {
        "students_compared": len(overlaps),
        "top_match_differs": top1_differs,
        "avg_shared_in_top_3": statistics.mean(overlaps),
        "identical_top_3": sum(o == 3 for o in overlaps)}

    # 4. Planted test pairs (sample data only)
    if args.sample and os.path.exists(paths["ground_truth"]):
        with open(paths["ground_truth"], encoding="utf-8") as f:
            planted = json.load(f)["planted_pairs"]
        results = []
        for p in planted:
            a, b, g = p["a"], p["b"], graphs
            if p["expect"] == "no_edge":
                passed = b not in g["ai"][a]
            elif p["expect"] == "best_match_both_modes":
                passed = all(g[m][a][b]["score"] == max(e["score"] for e in g[m][a].values()) for m in g)
            else:
                best_base = max(e["score"] for e in g["baseline"][a].values())
                best_ai = max(e["score"] for e in g["ai"][a].values())
                passed = g["baseline"][a][b]["score"] == best_base and g["ai"][a][b]["score"] < best_ai
            results.append((p["scenario"], passed))
        report["planted_tests"] = results

    print(json.dumps(report, indent=2))
    os.makedirs("results", exist_ok=True)
    name = "SYNTHETIC_system_report.json" if args.sample else "system_report.json"
    with open(os.path.join("results", name), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
