"""
How accurate is the LLM extraction? Compare what it extracted from the FAKE
students' text with the traits that text was written from.

    python parse_profiles.py --sample     # real LLM calls on the fake text (needs ANTHROPIC_API_KEY)
    python evaluate_extraction.py

Prints per-field accuracy and every disagreement, so you can read the text
and decide whether the model or the "truth" was wrong.
"""
import csv
import json

import config

paths = config.paths(sample=True)
with open(paths["ground_truth"], encoding="utf-8") as f:
    truth = json.load(f)["students"]
with open(paths["profiles"], encoding="utf-8") as f:
    data = json.load(f)
if data["meta"]["model"].startswith("offline"):
    raise SystemExit("profiles came from --offline mode. Run parse_profiles.py --sample (with the API) first.")
with open(paths["contacts"], newline="", encoding="utf-8") as f:
    email_of = {row["user_id"]: row["email"] for row in csv.DictReader(f)}

fields = ["social_energy", "study_habits", "cooking_frequency", "weekend_lifestyle", "habits", "extra_dealbreakers"]
correct = {field: 0 for field in fields}
mistakes = []
profiles = [p for p in data["profiles"] if p["extraction_status"] == "ok"]
for p in profiles:
    expected = truth[email_of[p["user_id"]]]
    for field in fields:
        got, want = p["traits"][field], expected[field]
        if isinstance(want, list):
            got, want = sorted(got), sorted(want)
        if got == want:
            correct[field] += 1
        else:
            mistakes.append(f"{p['user_id']} {field}: expected {want}, got {got}")

print(f"Accuracy over {len(profiles)} students ({data['meta']['model']}, prompt {data['meta']['prompt_version']})")
for field in fields:
    print(f"  {field:<20} {correct[field]:>3}/{len(profiles)}  ({100 * correct[field] / len(profiles):.0f}%)")
print(f"\n{len(mistakes)} disagreements:")
for line in mistakes:
    print("  " + line)
