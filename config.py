"""
MeetKat settings.

Every tunable number and every form option lives in this one file, so the
other scripts never hard-code them. To change how matching behaves, edit
WEIGHTS below and re-run matcher.py.
"""
import os

# ---------------------------------------------------------------- paths
DATA_DIR = "data"            # REAL student data. Gitignored. Never commit.
SAMPLE_DIR = "sample_data"   # FAKE data. Safe to commit.

# ---------------------------------------------------------------- LLM
# Whichever key is set in your terminal decides the provider (Claude is checked first).
PROVIDERS = {
    "anthropic": {
        "key_env_var": "ANTHROPIC_API_KEY",
        "model": "claude-haiku-4-5-20251001",      # small, cheap model: extraction is an easy task
        "url": "https://api.anthropic.com/v1/messages",
    },
    "gemini": {
        "key_env_var": "GEMINI_API_KEY",
        "model": "gemini-3.5-flash-lite",          # on Google's free tier (free tier data may be used by Google)
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    },
}
PROMPT_VERSION = "v1"                   # a label saved in the output files, for your records
MAX_API_ATTEMPTS = 3                    # retries for network / rate-limit errors (not bad JSON)

# ---------------------------------------------------------------- hard filters
MAX_MOVE_IN_GAP_DAYS = 42   # 6 weeks. With month answers, this means "same or adjacent month".
# Monthly budgets may miss each other by up to this much and still count as
# overlapping. Without it, "HK$4000–4999" and "HK$5000–5999" fail by $1.
BUDGET_TOLERANCE_HKD_MONTH = 500

# ---------------------------------------------------------------- scoring
# Relative importance of each part of the score. They don't need to add up to
# 100: the matcher divides by the total weight of the parts that apply.
WEIGHTS = {
    # structured answers (used by both modes)
    "cleanliness": 20,
    "guests": 10,
    "sleep": 15,
    "location": 15,
    "campus": 5,
    "cooking": 5,
    "languages": 5,
    # LLM-extracted traits ("ai" mode only)
    "social_energy": 10,
    "weekend": 5,
    "study_quiet": 10,
    "extra_dealbreakers": 20,
}
AI_ONLY_COMPONENTS = {"social_energy", "weekend", "study_quiet", "extra_dealbreakers"}
TOP_K = 10            # matches kept per student and mode (deep enough to find disagreements)
CONTESTED_DEPTH = 3   # a pick "belongs" to one algorithm only if the other didn't rank it in its top 3

# ---------------------------------------------------------------- experiment
EXPERIMENT_SEED = 2026   # makes the shuffle of the 3 matches reproducible

# Things a person might DO (habits) or refuse to live with (extra dealbreakers).
# The LLM must pick from this fixed list, so "A refuses X" can be checked
# against "B does X" with a simple set intersection.
LIFESTYLE_TAGS = {
    "parties": "hosts parties or pre-drinks at home",
    "overnight_guests": "has a partner or friends stay overnight often",
    "loud_music": "plays music, instruments or games loudly (no headphones)",
    "strong_food_smells": "often cooks strong-smelling food",
    "early_alarms": "gets up very early (before 6am) on most days",
    "leaves_dishes": "leaves dishes in the sink for a long time",
    "shoes_indoors": "wears shoes inside the home",
}

# ---------------------------------------------------------------- the form
# Order of columns in responses.csv.
COLUMNS = [
    "timestamp", "consent", "first_name", "email", "share_contact",
    "campus", "max_commute_min", "preferred_areas", "budget_bands", "move_in_month",
    "sleep_schedule", "cleanliness", "guests_frequency", "noise_tolerance",
    "smoker", "has_pet", "cooking", "languages", "dealbreakers",
    "ideal_flatmate_and_routine",
]

# The exact option text shown in the form -> the value our code uses.
# Copy the labels into Google Forms / Tally EXACTLY (and no commas in labels:
# the export joins multi-select answers with ", ").
FORM_OPTIONS = {
    "consent": {"I'm 18 or older and I agree to the above": "yes"},
    "share_contact": {"Yes": "yes", "No": "no"},
    "campus": {
        "HKU – Pok Fu Lam": "hku",
        "CUHK – Sha Tin": "cuhk",
        "HKUST – Clear Water Bay": "hkust",
        "PolyU – Hung Hom": "polyu",
        "CityU – Kowloon Tong": "cityu",
        "HKBU – Kowloon Tong": "hkbu",
        "EdUHK – Tai Po": "eduhk",
    },
    "max_commute_min": {
        "Up to 20 minutes": "20", "Up to 30 minutes": "30",
        "Up to 45 minutes": "45", "Up to 60 minutes": "60",
        "No preference": "none",
    },
    "preferred_areas": {
        "Western – Kennedy Town / Sai Ying Pun / Sheung Wan": "western",
        "Central & Wan Chai – Central / Wan Chai / Causeway Bay": "central",
        "Southern – Pok Fu Lam / Aberdeen": "southern",
        "Kowloon West – Mong Kok / Yau Ma Tei / Tsim Sha Tsui": "kowloon_west",
        "Kowloon Central – Kowloon Tong / Ho Man Tin / Hung Hom": "kowloon_central",
        "New Territories East – Sha Tin / Tai Po": "nt_east",
        "Sai Kung & Tseung Kwan O": "sai_kung",
        "No preference": "any",
    },
    "budget_bands": {
        "Under HK$4000": "0-3999", "HK$4000–4999": "4000-4999", "HK$5000–5999": "5000-5999",
        "HK$6000–7499": "6000-7499", "HK$7500 or more": "7500+",
    },
    "move_in_month": {
        "October 2026": "2026-10", "November 2026": "2026-11", "December 2026": "2026-12",
        "January 2027": "2027-01", "February 2027": "2027-02", "March 2027": "2027-03",
        "Flexible": "flexible",
    },
    "sleep_schedule": {
        "Early – usually asleep before 11pm": "early",
        "Middle – usually asleep 11pm–1am": "middle",
        "Late – often up past 1am": "late",
    },
    "smoker": {"No": "no", "Only outdoors": "outside", "Yes": "yes"},
    "has_pet": {"No": "no", "Yes": "yes"},
    "cooking": {
        "Rarely – mostly eat out or takeaway": "rarely",
        "A few times a week": "sometimes",
        "Most days": "most_days",
    },
    "languages": {
        name: name.lower() for name in [
            "English", "Cantonese", "Mandarin", "Korean", "Japanese", "Hindi",
            "Indonesian", "Vietnamese", "French", "German", "Spanish",
        ]
    },
    "dealbreakers": {
        "Someone who smokes or vapes (even outdoors)": "smoker",
        "Someone with a pet": "pets",
        "Someone who's usually up past 1am": "late_sleeper",
        "Someone who has guests over several times a week": "frequent_guests",
        "Someone messy": "messy",
        "None of these": "none",
    },
}
MULTI_SELECT = {"preferred_areas", "budget_bands", "languages", "dealbreakers"}
HAS_OTHER_BOX = {"campus", "languages"}   # free text allowed via "Other"
SCALE_1_TO_5 = {"cleanliness", "guests_frequency", "noise_tolerance"}

# Question text in a real export -> our column name. Our own column names
# also work, so a CSV can use either.
QUESTION_TITLES = {
    "Timestamp": "timestamp",
    "Submitted at": "timestamp",                     # Tally
    "Consent": "consent",
    "First name": "first_name",
    "Email": "email",
    "If you and a match both say yes, can we share your email with them?": "share_contact",
    "Which university do you attend?": "campus",
    "Which campus do you attend?": "campus",
    "Longest commute to campus you'd accept": "max_commute_min",
    "Areas you'd live in": "preferred_areas",
    "Monthly rent you'd pay for your room (HKD). Tick every range you'd consider.": "budget_bands",
    "When do you want to move in?": "move_in_month",
    "Sleep schedule": "sleep_schedule",
    "How tidy do you keep shared spaces?": "cleanliness",
    "How often do you have guests over?": "guests_frequency",
    "How much noise at home can you handle?": "noise_tolerance",
    "Do you smoke or vape?": "smoker",
    "Do you have a pet, or plan to bring one?": "has_pet",
    "How often do you cook at home?": "cooking",
    "Languages you speak comfortably": "languages",
    "Dealbreakers – I would NOT live with…": "dealbreakers",
    "Describe your ideal flatmate and how you live day to day.": "ideal_flatmate_and_routine",
}


def paths(sample=False):
    """All file locations in one place. sample=True switches every script to
    the fake files in sample_data/ (prefixed test_), so you can run the whole
    pipeline without touching real data:  python <script>.py --sample"""
    folder, prefix = (SAMPLE_DIR, "test_") if sample else (DATA_DIR, "")
    names = [
        "responses.csv", "profiles.json", "contacts.csv", "llm_cache.json",
        "rankings.json", "matches.json", "assignments.json", "access.json",
        "access_codes.csv", "streamlit_secrets.toml", "ratings.csv", "ground_truth.json",
    ]
    return {name.split(".")[0]: os.path.join(folder, prefix + name) for name in names}


def user_id_for_row(index):
    """Row 0 -> 'u001'. IDs follow the order of responses in the CSV."""
    return f"u{index + 1:03d}"


def label_for(column, value):
    """Reverse lookup: 'hku' -> 'HKU – Pok Fu Lam'."""
    for label, stored in FORM_OPTIONS.get(column, {}).items():
        if stored == value:
            return label
    return value
