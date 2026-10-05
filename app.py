"""
Step 5: the rating app.

    streamlit run app.py                      # real data in data/
    MEETKAT_SAMPLE=1 streamlit run app.py     # fake data in sample_data/ (try code from test_access_codes.csv)

A student opens their personal link (…/?code=K7M2QX) or types the code,
sees 3 matches and rates each one.

Privacy by design:
- The app never loads emails: matches.json and access.json contain none.
- The app never loads assignments.json, so it cannot reveal which algorithm
  chose a match, even by accident.
- On Streamlit Cloud the data comes from Secrets, never from the git repo.
"""
import csv
import html
import json
import os
import threading
from datetime import datetime, timezone

import streamlit as st

import config

SAMPLE = os.environ.get("MEETKAT_SAMPLE") == "1"
PATHS = config.paths(SAMPLE)
RATING_FIELDS = ["timestamp", "rater_id", "match_id", "slot", "rating", "would_message"]
_write_lock = threading.Lock()   # one Streamlit process serves every visitor: don't interleave writes

st.set_page_config(page_title="MeetKat · Your flatmate matches", page_icon="🏠", layout="centered")

st.markdown("""
<style>
  #MainMenu, footer {visibility: hidden;}
  .block-container {padding-top: 1.5rem; max-width: 640px;}
  .mk-header {display: flex; align-items: flex-end; gap: .6rem; flex-wrap: wrap;
              border-bottom: 3px solid #EE9A6A; padding-bottom: .6rem; margin-bottom: 1.2rem;}
  .mk-logo {font-size: 2rem; font-weight: 800; color: #3A3532; letter-spacing: -0.5px;}
  .mk-logo span {color: #D9692F;}
  .mk-tag {color: #6B7280; font-size: .95rem;}
  .mk-card {border: 1px solid #EADFD8; border-radius: 14px; padding: 1rem 1.1rem .6rem;
            margin: 1.4rem 0 .4rem; background: #FFF9F5;}
  .mk-name {font-size: 1.35rem; font-weight: 700; color: #3A3532; margin-bottom: .4rem;}
  .mk-chip {display: inline-block; background: #FBE3D3; color: #7A3A12; border-radius: 999px;
            padding: .15rem .65rem; margin: 0 .35rem .35rem 0; font-size: .85rem;}
  .mk-why {color: #3A3532; margin: .5rem 0 .3rem; line-height: 1.45;}
  @media (prefers-color-scheme: dark) {
    .mk-logo, .mk-name, .mk-why {color: #F3EEEA;}
    .mk-card {background: #2A2522; border-color: #4A403A;}
  }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------ data
def secret(name):
    """Read a Streamlit secret, or None. st.secrets raises an error when no
    secrets file exists (normal when running locally), so we catch it."""
    try:
        return st.secrets[name] if name in st.secrets else None
    except Exception:
        return None


@st.cache_data
def load(key):
    """'matches' -> MATCHES_JSON secret on the cloud, data/matches.json locally."""
    raw = secret(key.upper() + "_JSON")
    if raw:
        return json.loads(raw)
    with open(PATHS[key], encoding="utf-8") as f:
        return json.load(f)


def save_ratings(rows):
    """Streamlit Cloud's disk is wiped whenever the app restarts or sleeps, so
    on the cloud we append to a private Google Sheet. Locally: ratings.csv."""
    sheet_url = secret("RATINGS_SHEET_URL")
    if sheet_url:
        import gspread   # only needed on the cloud
        client = gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))
        client.open_by_url(sheet_url).sheet1.append_rows(
            [[row[field] for field in RATING_FIELDS] for row in rows], value_input_option="RAW")
        return
    with _write_lock:
        is_new = not os.path.exists(PATHS["ratings"])
        with open(PATHS["ratings"], "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RATING_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerows(rows)


# ------------------------------------------------------------ pages
def header():
    st.markdown('<div class="mk-header"><div class="mk-logo">Meet<span>Kat</span></div>'
                '<div class="mk-tag">Find your people, find your place</div></div>', unsafe_allow_html=True)


def login(access):
    """Return the user_id, or None if not logged in yet."""
    if st.session_state.get("user_id"):
        return st.session_state.user_id
    link_code = st.query_params.get("code", "").strip().upper()   # from the personal link
    if link_code in access:
        st.session_state.user_id = access[link_code]
        return st.session_state.user_id

    st.write("Enter the 6-character code from your MeetKat email to see your matches.")
    with st.form("login"):
        code = st.text_input("Your code", max_chars=6, placeholder="e.g. K7M2QX")
        if st.form_submit_button("See my matches", type="primary", width="stretch"):
            user_id = access.get(code.strip().upper())
            if user_id:
                st.session_state.user_id = user_id
                st.rerun()
            st.error("That code didn't work. Check your email and try again.")
    return None


def match_card(match):
    """All text is HTML-escaped: names and explanations came from outside our code."""
    chips = "".join(f'<span class="mk-chip">{html.escape(c)}</span>' for c in match["shared"])
    st.markdown(f'<div class="mk-card"><div class="mk-name">{html.escape(match["first_name"])}</div>'
                f'<div>{chips}</div><p class="mk-why">{html.escape(match["explanation"])}</p></div>',
                unsafe_allow_html=True)


def matches_page(user_id, user):
    st.subheader(f"Hi {user['first_name']}, meet 3 possible flatmates")
    if not user["matches"]:
        st.info("We couldn't find 3 good matches for you this round, mostly because of budget, "
                "move-in date or dealbreakers. Thanks for taking part. We'll be in touch if that changes.")
        return
    if st.session_state.get("submitted"):
        st.success("Thanks! Your ratings are saved. If you and a match both said yes to messaging "
                   "(and both agreed to share contacts), we'll introduce you by email.")
        return

    st.caption("They're in random order. Your ratings are anonymous and only used to improve matching.")
    with st.form("ratings"):
        answers = []
        for match in user["matches"]:
            match_card(match)
            rating = st.radio("How good a flatmate match is this person?", [1, 2, 3, 4, 5],
                              index=None, horizontal=True, key=f"rating_{match['slot']}",
                              help="1 = poor match, 5 = great match")
            message = st.radio("Would you want to message them?", ["Yes", "No"],
                               index=None, horizontal=True, key=f"message_{match['slot']}")
            answers.append((match, rating, message))
        submitted = st.form_submit_button("Submit my ratings", type="primary", width="stretch")

    if not submitted:
        return
    if any(rating is None or message is None for _, rating, message in answers):
        st.warning("Please answer both questions for all 3 people.")
        return
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [{"timestamp": now, "rater_id": user_id, "match_id": m["match_id"], "slot": m["slot"],
             "rating": rating, "would_message": message.lower()} for m, rating, message in answers]
    try:
        save_ratings(rows)
    except Exception as err:          # never show a stack trace to a student
        print(f"Saving ratings failed: {err}")
        st.error("Sorry, we couldn't save your answers. Please try again in a minute.")
        return
    st.session_state.submitted = True
    st.rerun()


def main():
    header()
    try:
        users, access = load("matches")["users"], load("access")
    except FileNotFoundError:
        st.error("No match data found. Run build_experiment.py first (or set MEETKAT_SAMPLE=1).")
        return
    user_id = login(access)
    if user_id and user_id in users:
        matches_page(user_id, users[user_id])
        if st.button("Not you? Use a different code"):
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()


main()
