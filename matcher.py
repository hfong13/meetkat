"""
Step 3: profiles.json  ->  rankings.json

    python matcher.py            # real data
    python matcher.py --sample   # fake data

For each mode ("baseline" = structured answers only, "ai" = structured
answers + LLM traits):
  1. Hard filters decide whether two students may be matched at all.
  2. Each surviving pair gets two DIRECTIONAL scores (how well B suits A, and
     how well A suits B), combined with a harmonic mean into one MUTUAL score.
  3. The pairs form a graph (adjacency dict): student -> {neighbour: edge}.
  4. A min-heap of size k picks each student's top k neighbours.

Complexity (n students, k = TOP_K):
  filters + scoring per pair ......... O(1)  (fixed number of answers and tags)
  building the graph ................. O(n^2) time, O(E) memory (E = edges <= n^2/2)
  top-k for one student with degree d  O(d log k)
  top-k for everyone ................. O(E log k) = O(n^2) since k is a constant
  whole run .......................... O(n^2)
"""
import argparse
import heapq
import json
import math
from datetime import date, datetime

import config

SLEEP_RANK = {"early": 0, "middle": 1, "late": 2}
COOKING_RANK = {"rarely": 0, "sometimes": 1, "most_days": 2}
SOCIAL_RANK = {"introvert": 0, "ambivert": 1, "extrovert": 2}
WEEKEND_RANK = {"homebody": 0, "balanced": 1, "out_a_lot": 2}


# =================================================================== 1. hard filters
# dealbreaker -> "does this OTHER person trigger it?"  (s = the other person's structured answers)
DEALBREAKER_RULES = {
    "smoker": lambda s: s["smoker"] != "no",               # includes "only outdoors"
    "pets": lambda s: s["has_pet"] == "yes",
    "late_sleeper": lambda s: s["sleep_schedule"] == "late",
    "frequent_guests": lambda s: s["guests_frequency"] >= 4,
    "messy": lambda s: s["cleanliness"] <= 2,
}


def budgets_overlap(a, b):
    """Two ranges overlap if each one starts before the other ends (plus a small
    tolerance, so neighbouring bands like HK$4000–4999 and HK$5000–5999 still match).
    A missing lower bound means 0; a missing upper bound ('HK$7500 or more') means no limit."""
    tolerance = config.BUDGET_TOLERANCE_HKD_MONTH
    a_low, b_low = a["budget_min_hkd_month"] or 0, b["budget_min_hkd_month"] or 0
    a_high = a["budget_max_hkd_month"] if a["budget_max_hkd_month"] is not None else math.inf
    b_high = b["budget_max_hkd_month"] if b["budget_max_hkd_month"] is not None else math.inf
    return a_low <= b_high + tolerance and b_low <= a_high + tolerance


def month_start(value):
    year, month = value.split("-")
    return date(int(year), int(month), 1)


def move_in_compatible(a, b):
    """'Flexible' fits anyone. Otherwise compare the first day of each month."""
    if "flexible" in (a["move_in_month"], b["move_in_month"]):
        return True
    gap = abs((month_start(a["move_in_month"]) - month_start(b["move_in_month"])).days)
    return gap <= config.MAX_MOVE_IN_GAP_DAYS


def violates_dealbreakers(chooser, other):
    """True if `other` triggers any of `chooser`'s dealbreakers."""
    return any(DEALBREAKER_RULES[d](other) for d in chooser["dealbreakers"] if d in DEALBREAKER_RULES)


def passes_hard_filters(a, b):
    """a and b are structured answers. Dealbreakers are checked BOTH ways."""
    return (budgets_overlap(a, b)
            and move_in_compatible(a, b)
            and not violates_dealbreakers(a, b)
            and not violates_dealbreakers(b, a))


# =================================================================== 2. scoring
# Every component answers "how happy would A be living with B?" on a 0-1 scale.
# None means "doesn't apply or unknown" and the component is skipped, so
# missing information is neither rewarded nor punished.
#
# Directional components follow one rule:
#     1 - (how much WORSE B is than A on this) x (how much A cares)
# so B is only penalised for being worse than A, and only if A minds.

def similarity(rank_a, rank_b, max_gap):
    """Symmetric: 1 when equal, 0 at opposite ends of the scale."""
    return 1 - abs(rank_a - rank_b) / max_gap


def trait_similarity(value_a, value_b, ranks):
    if "unknown" in (value_a, value_b):
        return None
    return similarity(ranks[value_a], ranks[value_b], max(ranks.values()))


def areas_overlap(a, b):
    return "any" in a["preferred_areas"] or "any" in b["preferred_areas"] \
        or bool(set(a["preferred_areas"]) & set(b["preferred_areas"]))


def jaccard(set_a, set_b):
    """Shared items / all items: {english, mandarin} vs {english} -> 1/2."""
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else None


def noise_level(profile, use_social):
    """0 (quiet) to 1 (lively), from how often they have guests
    and, if we know it for both people, their social energy."""
    level = (profile["structured"]["guests_frequency"] - 1) / 4
    if use_social:
        level = (level + SOCIAL_RANK[profile["traits"]["social_energy"]] / 2) / 2
    return level


def study_quiet_score(a, b):
    """Only matters if A studies at home and needs quiet."""
    if a["traits"]["study_habits"] != "home_needs_quiet":
        return None
    use_social = "unknown" not in (a["traits"]["social_energy"], b["traits"]["social_energy"])
    return 1 - max(0, noise_level(b, use_social) - noise_level(a, use_social))


def extra_dealbreaker_score(traits_a, traits_b):
    """Soft penalty, not a hard filter: these come from the LLM, which can be
    wrong, so a misread sentence lowers a score instead of hiding a match."""
    if not traits_a["extra_dealbreakers"]:
        return None
    clash = set(traits_a["extra_dealbreakers"]) & set(traits_b["habits"])
    return 0.0 if clash else 1.0


def component_scores(a, b, mode):
    """A's view of B, one 0-1 score (or None) per component."""
    sa, sb = a["structured"], b["structured"]
    sensitivity = (5 - sa["noise_tolerance"]) / 4        # 0 = unbothered, 1 = very sensitive

    scores = {
        # directional
        "cleanliness": 1 - max(0, sa["cleanliness"] - sb["cleanliness"]) / 4,
        "guests": 1 - max(0, sb["guests_frequency"] - sa["guests_frequency"]) / 4 * sensitivity,
        "sleep": 1 - max(0, SLEEP_RANK[sb["sleep_schedule"]] - SLEEP_RANK[sa["sleep_schedule"]]) / 2 * sensitivity,
        # symmetric
        "location": 1.0 if areas_overlap(sa, sb) else 0.0,
        "campus": None if not sa["campus"] or not sb["campus"] else float(sa["campus"] == sb["campus"]),
        "cooking": similarity(COOKING_RANK[sa["cooking"]], COOKING_RANK[sb["cooking"]], 2),
        "languages": jaccard(set(sa["languages"]), set(sb["languages"])),
    }
    if mode == "ai":
        ta, tb = a["traits"], b["traits"]
        scores["social_energy"] = trait_similarity(ta["social_energy"], tb["social_energy"], SOCIAL_RANK)
        scores["weekend"] = trait_similarity(ta["weekend_lifestyle"], tb["weekend_lifestyle"], WEEKEND_RANK)
        scores["study_quiet"] = study_quiet_score(a, b)
        scores["extra_dealbreakers"] = extra_dealbreaker_score(ta, tb)
    return scores


def directional_score(a, b, mode):
    """0-100: how well B suits A. A weighted average over components that apply.
    Returns (score, the component scores that were used)."""
    used = {name: s for name, s in component_scores(a, b, mode).items() if s is not None}
    total_weight = sum(config.WEIGHTS[name] for name in used)
    weighted = sum(config.WEIGHTS[name] * s for name, s in used.items())
    return 100 * weighted / total_weight, used


def harmonic_mean(x, y):
    """Pulled towards the LOWER score: 90 and 30 give 45 (a plain average gives 60).
    A match that only works for one person scores low."""
    return 0.0 if x + y == 0 else 2 * x * y / (x + y)


# =================================================================== 3. the graph
def build_graph(profiles, mode):
    """graph[a][b] = edge. Only pairs that pass the hard filters get an edge.
    Each pair is checked once (j > i) and stored in both directions."""
    graph = {p["user_id"]: {} for p in profiles}
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            a, b = profiles[i], profiles[j]
            if not passes_hard_filters(a["structured"], b["structured"]):
                continue
            a_view, a_parts = directional_score(a, b, mode)
            b_view, b_parts = directional_score(b, a, mode)
            mutual = harmonic_mean(a_view, b_view)
            graph[a["user_id"]][b["user_id"]] = {"score": mutual, "my_view": a_view,
                                                 "their_view": b_view, "components": a_parts}
            graph[b["user_id"]][a["user_id"]] = {"score": mutual, "my_view": b_view,
                                                 "their_view": a_view, "components": b_parts}
    return graph


# =================================================================== 4. top-k with a heap
def top_k(neighbours, k, order):
    """Keep a MIN-heap of the best k seen so far. The worst of those k sits at
    heap[0], so each new candidate is compared to it in O(1) and swapped in
    O(log k). Total O(d log k) instead of O(d log d) for sorting everyone.

    Ties on score go to whoever answered the form first (smaller row order),
    so results are reproducible. Tuples compare element by element, and
    -order makes an earlier row count as 'bigger'."""
    heap = []
    for other_id, edge in neighbours.items():
        item = (edge["score"], -order[other_id], other_id)
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)      # pop the smallest and push the new one
    return [other_id for _, _, other_id in sorted(heap, reverse=True)]


def rank_all(profiles, mode, k=config.TOP_K):
    graph = build_graph(profiles, mode)
    order = {p["user_id"]: i for i, p in enumerate(profiles)}
    names = {p["user_id"]: p["first_name"] for p in profiles}
    results = {}
    for user_id, neighbours in graph.items():
        results[user_id] = {
            "eligible_count": len(neighbours),
            "top": [
                {
                    "user_id": other,
                    "first_name": names[other],
                    "score": round(neighbours[other]["score"], 1),
                    "my_view": round(neighbours[other]["my_view"], 1),
                    "their_view": round(neighbours[other]["their_view"], 1),
                    "components": {c: round(v, 2) for c, v in neighbours[other]["components"].items()},
                }
                for other in top_k(neighbours, k, order)
            ],
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Rank flatmate matches in both modes.")
    parser.add_argument("--sample", action="store_true", help="use the fake files in sample_data/")
    args = parser.parse_args()
    paths = config.paths(args.sample)

    with open(paths["profiles"], encoding="utf-8") as f:
        profiles = json.load(f)["profiles"]
    ai, baseline = rank_all(profiles, "ai"), rank_all(profiles, "baseline")

    users = {
        p["user_id"]: {
            "first_name": p["first_name"],
            "eligible_count": ai[p["user_id"]]["eligible_count"],   # same filters in both modes
            "ai": ai[p["user_id"]]["top"],
            "baseline": baseline[p["user_id"]]["top"],
        }
        for p in profiles
    }
    output = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "weights": config.WEIGHTS,
            "ai_only_components": sorted(config.AI_ONLY_COMPONENTS),
            "max_move_in_gap_days": config.MAX_MOVE_IN_GAP_DAYS,
            "top_k": config.TOP_K,
        },
        "users": users,
    }
    with open(paths["rankings"], "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    few = [u for u, v in users.items() if v["eligible_count"] < 3]
    print(f"Ranked {len(profiles)} students -> {paths['rankings']}")
    if few:
        print(f"  {len(few)} student(s) have fewer than 3 possible matches: {', '.join(few)}")


if __name__ == "__main__":
    main()
