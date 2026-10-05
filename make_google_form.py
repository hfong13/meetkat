"""
Generate a Google Apps Script that builds the MeetKat intake form.

    python make_google_form.py        -> writes form/create_form.gs

Every question title and option label is copied from config.py, so the
form's CSV export always matches what parse_profiles.py expects.
How to use the .gs file: see the comment at the top of it.
"""
import json
import os

import config

CONTACT_EMAIL = "your.email@example.com"   # shown in the consent text: where to ask for deletion
DELETE_BY = "31 January 2027"              # date you promise to delete all data by

T = {v: k for k, v in config.QUESTION_TITLES.items()
     if k not in ("Timestamp", "Submitted at", "Which campus do you attend?")}


def labels(column, drop=()):
    return [label for label, value in config.FORM_OPTIONS[column].items() if value not in drop]


INTRO = ("A student project matching university students in Hong Kong with compatible flatmates. "
         "Takes about 3 minutes.\n\n"
         "What happens next: in about a week you'll get an email with a private link to 3 suggested "
         "flatmates. You rate each one (1 minute). You don't have to be looking for a flatmate right now: "
         "just answer as if you were.")

CONSENT = (
    "Before you start:\n"
    "• Your answers are used ONLY to suggest flatmate matches and for anonymous research on how well "
    "the matching works.\n"
    "• Your written answer is processed by an AI service (Anthropic's Claude API) to understand your "
    "preferences. Your name is removed first.\n"
    "• Matches see only your FIRST NAME and a short note about what you have in common. Never your "
    "email or surname.\n"
    "• Your email is shared with a match ONLY if you and they both agree.\n"
    "• MeetKat doesn't verify anyone's identity. If you meet someone, meet in a public place first.\n"
    f"• To delete your data at any time, email {CONTACT_EMAIL}. All data is deleted by {DELETE_BY}."
)

# (kind, column, extra)  kinds: section, checkbox_consent, text, email, choice, checkboxes, dropdown, scale, paragraph
QUESTIONS = [
    ("section", "Consent", None),
    ("consent", "consent", None),
    ("section", "About you", None),
    ("text", "first_name", "Only your first name is shown to matches."),
    ("email", "email", "Used only to send your private match link. Never shown to anyone."),
    ("choice", "share_contact", None),
    ("section", "Your place", None),
    ("choice_other", "campus", None),
    ("choice", "max_commute_min", None),
    ("checkboxes", "preferred_areas", None),
    ("checkboxes", "budget_bands", None),
    ("dropdown", "move_in_month", None),
    ("section", "How you live", None),
    ("choice", "sleep_schedule", None),
    ("scale", "cleanliness", ("Relaxed – a bit of mess is fine", "Spotless – clean up straight away")),
    ("scale", "guests_frequency", ("Almost never", "Most days")),
    ("scale", "noise_tolerance", ("I need it quiet", "Noise doesn't bother me")),
    ("choice", "smoker", None),
    ("choice", "has_pet", None),
    ("choice", "cooking", None),
    ("checkboxes_other", "languages", None),
    ("checkboxes", "dealbreakers", None),
    ("paragraph", "ideal_flatmate_and_routine",
     "For example: your typical weekday, how social you want home to be, what a great flatmate does, "
     "and anything that annoys you. A few sentences is perfect (at least 100 characters)."),
]


def js(value):
    return json.dumps(value, ensure_ascii=False)


def build():
    lines = []
    for kind, column, extra in QUESTIONS:
        if kind == "section":
            lines.append(f"  form.addPageBreakItem().setTitle({js(column)});")
            continue
        title = js(T[column])
        if kind == "consent":
            lines.append(f"  form.addCheckboxItem().setTitle({title}).setHelpText({js(CONSENT)})"
                         f".setChoiceValues({js(labels('consent'))}).setRequired(true);")
        elif kind == "text":
            lines.append(f"  form.addTextItem().setTitle({title}).setHelpText({js(extra)}).setRequired(true);")
        elif kind == "email":
            lines.append(f"  form.addTextItem().setTitle({title}).setHelpText({js(extra)}).setRequired(true)\n"
                         f"      .setValidation(FormApp.createTextValidation().requireTextIsEmail()"
                         f".setHelpText('Please enter a valid email.').build());")
        elif kind in ("choice", "choice_other"):
            other = ".showOtherOption(true)" if kind == "choice_other" else ""
            lines.append(f"  form.addMultipleChoiceItem().setTitle({title})"
                         f".setChoiceValues({js(labels(column))}){other}.setRequired(true);")
        elif kind in ("checkboxes", "checkboxes_other"):
            other = ".showOtherOption(true)" if kind == "checkboxes_other" else ""
            lines.append(f"  form.addCheckboxItem().setTitle({title})"
                         f".setChoiceValues({js(labels(column))}){other}.setRequired(true);")
        elif kind == "dropdown":
            lines.append(f"  form.addListItem().setTitle({title})"
                         f".setChoiceValues({js(labels(column))}).setRequired(true);")
        elif kind == "scale":
            low, high = extra
            lines.append(f"  form.addScaleItem().setTitle({title}).setBounds(1, 5)"
                         f".setLabels({js(low)}, {js(high)}).setRequired(true);")
        elif kind == "paragraph":
            lines.append(f"  form.addParagraphTextItem().setTitle({title}).setHelpText({js(extra)}).setRequired(true)\n"
                         f"      .setValidation(FormApp.createParagraphTextValidation()"
                         f".requireTextLengthGreaterThanOrEqualTo(100)"
                         f".setHelpText('Please write at least 100 characters.').build());")
    body = "\n".join(lines)
    return f"""/**
 * MeetKat intake form builder. GENERATED by make_google_form.py: edit config.py, not this file.
 *
 * HOW TO USE (about 2 minutes):
 *  1. Go to https://script.google.com, signed in to the Google account that should own the form.
 *  2. Click "New project". Delete the sample code and paste this whole file.
 *  3. Click Save, then Run (function: createMeetKatForm).
 *  4. Approve the permissions Google asks for (it needs to create a form and a sheet in YOUR Drive).
 *  5. Open "Execution log": it prints the link to share, the edit link and the responses sheet.
 *
 * IMPORTANT: once responses arrive, don't edit question titles or option labels in the form.
 * The code matches answers by their exact text.
 */
function createMeetKatForm() {{
  var form = FormApp.create('MeetKat – find a flatmate who fits how you live');
  form.setDescription({js(INTRO)});
  form.setCollectEmail(false);          // we ask for email as a question instead
  form.setLimitOneResponsePerUser(false); // true would force students to sign in to Google
  form.setProgressBar(true);
  form.setConfirmationMessage("Thanks! In about a week you'll get an email with a private link to 3 suggested flatmates. It takes 1 minute to rate them.");

{body}

  var sheet = SpreadsheetApp.create('MeetKat responses');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, sheet.getId());

  Logger.log('SHARE THIS LINK WITH STUDENTS: ' + form.getPublishedUrl());
  Logger.log('Edit the form: ' + form.getEditUrl());
  Logger.log('Responses sheet: ' + sheet.getUrl());
}}
"""


def main():
    os.makedirs("form", exist_ok=True)
    with open("form/create_form.gs", "w", encoding="utf-8") as f:
        f.write(build())
    print("Wrote form/create_form.gs")


if __name__ == "__main__":
    main()
