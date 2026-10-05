"""
Make SIMULATED ratings for the fake students, so you can test analyze.py and
see the chart before real data exists. Sample data only: it refuses to
touch data/.

    python simulate_ratings.py
    python analyze.py --sample

The fake rating is the matcher's AI-mode score turned into 1-5, plus noise.
That builds in an AI advantage on purpose, so the numbers mean nothing
about the real world. They only prove the analysis code runs.
"""
import csv
import json
import random
from datetime import datetime, timedelta

import config
import matcher

paths = config.paths(sample=True)
rng = random.Random(3)

with open(paths["profiles"], encoding="utf-8") as f:
    profiles = {p["user_id"]: p for p in json.load(f)["profiles"]}
with open(paths["matches"], encoding="utf-8") as f:
    users = json.load(f)["users"]

rows, start = [], datetime(2026, 11, 1, 10, 0)
for rater, info in users.items():
    for m in info["matches"]:
        score, _ = matcher.directional_score(profiles[rater], profiles[m["match_id"]], "ai")
        rating = min(5, max(1, round(score / 20 + rng.gauss(0, 0.9))))
        rows.append({"timestamp": (start + timedelta(minutes=rng.randint(0, 5000))).isoformat(timespec="seconds"),
                     "rater_id": rater, "match_id": m["match_id"], "slot": m["slot"], "rating": rating,
                     "would_message": "yes" if rating + rng.gauss(0, 1) >= 3.5 else "no"})

with open(paths["ratings"], "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["timestamp", "rater_id", "match_id", "slot", "rating", "would_message"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} SIMULATED ratings to {paths['ratings']}")
