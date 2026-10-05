"""
Create 30 FAKE students so the pipeline can be tested without real data.

    python generate_test_data.py

Writes:
  sample_data/test_responses.csv     same columns and option labels as the real form export
  sample_data/test_ground_truth.json the traits each free-text answer was written FROM,
                                     plus the planted test pairs

Because every free-text answer is built from known traits, we know the right
answer for the LLM extraction and can measure its accuracy. Traits that a
student's text doesn't mention are recorded as "unknown": the LLM can't know
them either.

Planted pairs (so the matcher has known right answers):
  twins_1, twins_2    identical answers, different wording -> each other's best match
  smoker_conflict     twins, but one smokes outdoors and the other's dealbreaker is smoking -> no edge
  budget_conflict     twins with budgets that don't overlap -> no edge
  move_in_conflict    twins moving in 3 months apart -> no edge
  pet_conflict        one has a pet, the other's dealbreaker is pets -> no edge
  hidden_conflict     identical structured answers, but the free text says one hosts
                      parties and the other can't stand them -> baseline loves it, AI shouldn't
"""
import copy
import csv
import json
import os
import random
import re
from datetime import datetime, timedelta

import config
from matcher import DEALBREAKER_RULES

SEED = 7            # same seed -> same fake data every time
N_STUDENTS = 30

NAMES = ["Aarav", "Mei", "Lucas", "Priya", "Minh", "Sofia", "Jae-won", "Olivia", "Rahul",
         "Chloe", "Kenji", "Amira", "Tom", "Ana", "Wei", "Hannah", "Sanjana", "Diego",
         "Yuki", "Liam", "Nadia", "Bao", "Emma", "Arjun", "Fatima", "Jack", "Lin",
         "Isabella", "Kiran", "Grace"]

AREAS = [v for v in config.FORM_OPTIONS["preferred_areas"].values() if v != "any"]
BANDS = list(config.FORM_OPTIONS["budget_bands"].values())
OTHER_LANGUAGES = [v for v in config.FORM_OPTIONS["languages"].values() if v != "english"]
DEALBREAKERS = [v for v in config.FORM_OPTIONS["dealbreakers"].values() if v != "none"]
TAGS = sorted(config.LIFESTYLE_TAGS)

# ---------------------------------------------------------------- sentence banks
SOCIAL_TEXT = {
    "introvert": ["I'm pretty introverted and need my own space to recharge after uni.",
                  "Honestly I'm on the quiet side, I like having calm evenings to myself.",
                  "I'm a bit shy at first and I value a peaceful home more than a social one."],
    "ambivert": ["I'm social but I also like my alone time, so a mix is perfect for me.",
                 "Happy to chat over dinner or watch something together, but I don't need to be best friends with my flatmates.",
                 "I'm somewhere in the middle: friendly, but I respect people's space."],
    "extrovert": ["I'm very outgoing and would love flatmates I can hang out with.",
                  "I'm a people person! Would be great to cook together or go hiking as a flat.",
                  "I get my energy from being around people, so a social flat would be ideal."],
}
STUDY_TEXT = {
    "home_needs_quiet": ["I study at home most days and really need it quiet when I do.",
                         "Most of my study happens at my desk at home, so noise in the evenings is hard for me.",
                         "I have heaps of readings and I work best in a silent room at home."],
    "home_flexible": ["I study at home a lot but background noise is fine, I just put headphones on.",
                      "I do most assignments at home and noise doesn't really bother me."],
    "mostly_on_campus": ["I mostly study at the library so I'm not home much on weekdays.",
                         "Uni days are long for me, I'm usually on campus until the evening.",
                         "I do all my study on campus, home is just for relaxing."],
}
WEEKEND_TEXT = {
    "homebody": ["Weekends I like to stay in, cook and watch series.",
                 "On weekends I'm usually home reading or gaming."],
    "balanced": ["Weekends are a mix: brunch with friends, then a quiet night in.",
                 "I go out sometimes on weekends but I'm just as happy at home."],
    "out_a_lot": ["I'm out most weekends, hiking Dragon's Back, junk trips, going out in Central at night.",
                  "Weekends I'm hardly home, always exploring Hong Kong or out late with friends."],
}
COOKING_TEXT = {
    "rarely": ["I don't really cook, mostly takeaway or eating on campus.",
               "Cooking isn't my thing, I usually grab food out."],
    "sometimes": ["I cook a few times a week, nothing fancy.",
                  "I cook maybe three nights a week and eat out the rest."],
    "most_days": ["I cook almost every day and love trying new recipes.",
                  "I cook dinner most nights, so I use the kitchen a fair bit."],
}
HABIT_TEXT = {
    "parties": "I like hosting pre-drinks or small parties at home now and then.",
    "overnight_guests": "My partner stays over a couple of nights a week.",
    "loud_music": "I play guitar and listen to music on speakers quite a lot.",
    "strong_food_smells": "I cook a lot of spicy food with strong smells, just a heads up haha.",
    "early_alarms": "I'm up at 5:30 most mornings for the gym.",
    "leaves_dishes": "I'm not perfect with dishes, sometimes they sit until the next day.",
    "shoes_indoors": "I usually keep my shoes on inside.",
}
DEALBREAKER_TEXT = {
    "parties": "I really can't live somewhere with parties at home.",
    "overnight_guests": "I'd prefer no partners staying over regularly.",
    "loud_music": "Loud music through speakers would drive me crazy.",
    "strong_food_smells": "Strong cooking smells bother me a lot.",
    "early_alarms": "Please no super early alarms, I'm a light sleeper in the morning.",
    "leaves_dishes": "Dishes left in the sink for days is my biggest pet peeve.",
    "shoes_indoors": "I'd really like a shoes-off household.",
}
OPENERS = ["", "Hi! ", "Hey, ", "Hello, I'm {name}. ", "Hi there! "]
BACKGROUNDS = ["I'm an exchange student from Germany here for one semester.",
               "Master's student in finance, just moved from Shanghai.",
               "Local student from Tuen Mun, moving out for the first time.",
               "First year in Hong Kong, studying engineering.",
               "I'm from Seoul doing a Master of Data Science.",
               "PhD student in marine science, originally from Chile.",
               "Exchange student from Toronto, here for spring semester.",
               "Final-year architecture student from Jakarta.",
               "I'm from Mumbai, doing an MBA.",
               "Second-year law student from Singapore."]
IDEALS = ["Ideally my flatmate is respectful and says so if something bothers them.",
          "Looking for someone reliable who pays bills on time and is easy to talk to.",
          "I'd love someone chill and honest.",
          "Mostly I just want someone considerate about shared spaces.",
          "Would be nice to live with someone open to different food and cultures."]
CLOSERS = ["", " Thanks!", " Looking forward to meeting people :)", " Happy to chat more!"]


# ---------------------------------------------------------------- one random student
def random_budget(rng):
    start = rng.choices(range(len(BANDS)), weights=[1, 4, 5, 3, 1])[0]   # most around HK$5000-5999
    return BANDS[start:start + rng.choice([1, 2, 2])]


def make_consistent(s):
    """Nobody should hold a dealbreaker that they themselves trigger."""
    s["dealbreakers"] = [d for d in s["dealbreakers"] if not DEALBREAKER_RULES[d](s)]
    s["extra_dealbreakers"] = [t for t in s["extra_dealbreakers"] if t not in s["habits"]]


def random_student(rng):
    social = rng.choice(["introvert", "ambivert", "extrovert"])
    tag_a, tag_b = rng.sample(TAGS, 2)
    s = {
        "campus": rng.choice(["hku", "hku", "hku", "cuhk", "hkust", "polyu", "cityu", "hkbu"]),
        "max_commute_min": rng.choice(["20", "30", "30", "45", "60", "none"]),
        "preferred_areas": ["any"] if rng.random() < 0.15 else rng.sample(AREAS, rng.randint(1, 3)),
        "budget_bands": random_budget(rng),
        "move_in_month": rng.choice(["2026-11", "2026-12", "2027-01", "2027-02", "2027-02", "flexible"]),
        "sleep_schedule": rng.choice(["early", "middle", "middle", "late"]),
        "cleanliness": rng.randint(2, 5),
        # more social people tend to have more guests
        "guests_frequency": {"introvert": rng.randint(1, 2), "ambivert": rng.randint(2, 3),
                             "extrovert": rng.randint(3, 5)}[social],
        "noise_tolerance": rng.randint(1, 5),
        "smoker": rng.choices(["no", "outside", "yes"], weights=[85, 10, 5])[0],
        "has_pet": "yes" if rng.random() < 0.1 else "no",
        "cooking": rng.choice(["rarely", "sometimes", "most_days"]),
        "languages": ["english"] + rng.sample(OTHER_LANGUAGES, rng.choice([0, 1, 1, 2])),
        "dealbreakers": rng.sample(DEALBREAKERS, rng.choice([0, 1, 1, 2])),
        "share_contact": "yes" if rng.random() < 0.8 else "no",
        # traits the free text is written from
        "social_energy": social,
        "study_habits": rng.choice(["home_needs_quiet", "home_flexible", "mostly_on_campus"]),
        "weekend_lifestyle": rng.choice(["homebody", "balanced", "out_a_lot"]),
        "habits": [tag_a] if rng.random() < 0.4 else [],
        "extra_dealbreakers": [tag_b] if rng.random() < 0.4 else [],
    }
    make_consistent(s)
    return s


def write_free_text(s, name, rng):
    """Return (text, ground truth). Each trait is mentioned with probability
    0.8; unmentioned ones are 'unknown' in the ground truth."""
    truth = {"habits": sorted(s["habits"]), "extra_dealbreakers": sorted(s["extra_dealbreakers"])}
    sentences = []
    for field, value, bank in [("social_energy", s["social_energy"], SOCIAL_TEXT),
                               ("study_habits", s["study_habits"], STUDY_TEXT),
                               ("weekend_lifestyle", s["weekend_lifestyle"], WEEKEND_TEXT),
                               ("cooking_frequency", s["cooking"], COOKING_TEXT)]:
        if rng.random() < 0.8:
            sentences.append(rng.choice(bank[value]))
            truth[field] = value
        else:
            truth[field] = "unknown"
    sentences += [HABIT_TEXT[t] for t in s["habits"]]
    sentences += [DEALBREAKER_TEXT[t] for t in s["extra_dealbreakers"]]
    rng.shuffle(sentences)
    text = (rng.choice(OPENERS).format(name=name) + rng.choice(BACKGROUNDS) + " "
            + " ".join(sentences) + " " + rng.choice(IDEALS) + rng.choice(CLOSERS))
    return text.strip(), truth


# ---------------------------------------------------------------- planted pairs
def planted_students(rng):
    """Return (students, scenarios). Scenario a/b are positions in `students`."""
    students, scenarios = [], []

    def add(scenario, expect, a, b):
        make_consistent(a)
        make_consistent(b)
        students.extend([a, b])
        scenarios.append({"scenario": scenario, "expect": expect,
                          "a": len(students) - 2, "b": len(students) - 1})

    for name in ["twins_1", "twins_2"]:
        a = random_student(rng)
        add(name, "best_match_both_modes", a, copy.deepcopy(a))

    a = random_student(rng)
    a["smoker"], a["dealbreakers"] = "no", ["smoker"]
    b = copy.deepcopy(a)
    b["smoker"], b["dealbreakers"] = "outside", []
    add("smoker_conflict", "no_edge", a, b)

    a = random_student(rng)
    a["budget_bands"] = ["4000-4999"]
    b = copy.deepcopy(a)
    b["budget_bands"] = ["6000-7499"]
    add("budget_conflict", "no_edge", a, b)

    a = random_student(rng)
    a["move_in_month"] = "2026-11"
    b = copy.deepcopy(a)
    b["move_in_month"] = "2027-02"
    add("move_in_conflict", "no_edge", a, b)

    a = random_student(rng)
    a["has_pet"], a["dealbreakers"] = "yes", []
    b = copy.deepcopy(a)
    b["has_pet"], b["dealbreakers"] = "no", ["pets"]
    add("pet_conflict", "no_edge", a, b)

    a = random_student(rng)
    a["habits"], a["extra_dealbreakers"] = [], ["parties"]
    a["social_energy"], a["weekend_lifestyle"] = "introvert", "homebody"
    b = copy.deepcopy(a)
    b["habits"], b["extra_dealbreakers"] = ["parties"], []
    b["social_energy"], b["weekend_lifestyle"] = "extrovert", "out_a_lot"
    add("hidden_conflict", "baseline_best_but_ai_lower", a, b)
    return students, scenarios


# ---------------------------------------------------------------- CSV row
def label(column, value):
    return config.label_for(column, value)


def to_row(s, name, email, text, timestamp):
    return {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "consent": label("consent", "yes"),
        "first_name": name,
        "email": email,
        "share_contact": label("share_contact", s["share_contact"]),
        "campus": label("campus", s["campus"]),
        "max_commute_min": label("max_commute_min", s["max_commute_min"]),
        "preferred_areas": ", ".join(label("preferred_areas", a) for a in s["preferred_areas"]),
        "budget_bands": ", ".join(label("budget_bands", b) for b in s["budget_bands"]),
        "move_in_month": label("move_in_month", s["move_in_month"]),
        "sleep_schedule": label("sleep_schedule", s["sleep_schedule"]),
        "cleanliness": s["cleanliness"],
        "guests_frequency": s["guests_frequency"],
        "noise_tolerance": s["noise_tolerance"],
        "smoker": label("smoker", s["smoker"]),
        "has_pet": label("has_pet", s["has_pet"]),
        "cooking": label("cooking", s["cooking"]),
        "languages": ", ".join(label("languages", l) for l in s["languages"]),
        "dealbreakers": ", ".join(label("dealbreakers", d) for d in s["dealbreakers"])
                        or label("dealbreakers", "none"),
        "ideal_flatmate_and_routine": text,
    }


def main():
    rng = random.Random(SEED)
    students, scenarios = planted_students(rng)
    while len(students) < N_STUDENTS:
        students.append(random_student(rng))

    # Shuffle so planted pairs aren't next to each other, then remember where they went.
    order = list(range(N_STUDENTS))
    rng.shuffle(order)                        # order[new_position] = old_position
    new_position = {old: new for new, old in enumerate(order)}
    names = rng.sample(NAMES, N_STUDENTS)
    paths = config.paths(sample=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)

    rows, truth = [], {}
    when = datetime(2026, 10, 1, 9, 0)
    for position, old in enumerate(order):
        s, name = students[old], names[position]
        email = f"{re.sub(r'[^a-z]', '', name.lower())}.{position + 1}@example.com"
        text, truth[email] = write_free_text(s, name, rng)
        when += timedelta(minutes=rng.randint(5, 300))
        rows.append(to_row(s, name, email, text, when))

    with open(paths["responses"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    planted = [{"scenario": sc["scenario"], "expect": sc["expect"],
                "a": config.user_id_for_row(new_position[sc["a"]]),
                "b": config.user_id_for_row(new_position[sc["b"]])} for sc in scenarios]
    with open(paths["ground_truth"], "w", encoding="utf-8") as f:
        json.dump({"note": "FAKE data. Traits each free-text answer was generated from.",
                   "students": truth, "planted_pairs": planted}, f, indent=2)

    print(f"Wrote {N_STUDENTS} fake students to {paths['responses']}")
    print(f"Wrote the known answers and planted pairs to {paths['ground_truth']}")
    for p in planted:
        print(f"  {p['scenario']:<17} {p['a']} + {p['b']}  expect: {p['expect']}")


if __name__ == "__main__":
    main()
