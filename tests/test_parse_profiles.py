"""Tests for parse_profiles.py and llm.py, using a FAKE API (no key, no cost)."""
import json
import os
import tempfile
import unittest

import llm
import parse_profiles as pp

GOOD = {"social_energy": "introvert", "study_habits": "home_needs_quiet", "cooking_frequency": "most_days",
        "weekend_lifestyle": "homebody", "habits": [], "extra_dealbreakers": ["parties"]}


class FakeAPI:
    """Replays prepared replies and records every call it receives."""
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, system, messages, json_schema=None):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


class ValidationTests(unittest.TestCase):
    def test_valid_reply(self):
        traits, errors = pp.validate_traits(json.dumps(GOOD))
        self.assertEqual(errors, [])
        self.assertEqual(traits["extra_dealbreakers"], ["parties"])

    def test_capitalised_enum_is_accepted(self):
        traits, errors = pp.validate_traits(json.dumps({**GOOD, "social_energy": "Introvert"}))
        self.assertEqual(errors, [])
        self.assertEqual(traits["social_energy"], "introvert")

    def test_bad_value_missing_field_extra_field_and_bad_json(self):
        bad = {**GOOD, "social_energy": "shy", "mood": "happy"}
        del bad["habits"]
        _, errors = pp.validate_traits(json.dumps(bad))
        self.assertEqual(len(errors), 3)
        _, errors = pp.validate_traits("{not json")
        self.assertIn("not valid JSON", errors[0])

    def test_unknown_tag_and_contradiction(self):
        _, errors = pp.validate_traits(json.dumps({**GOOD, "habits": ["juggling"]}))
        self.assertTrue(any("unknown tags" in e for e in errors))
        _, errors = pp.validate_traits(json.dumps({**GOOD, "habits": ["parties"]}))
        self.assertTrue(any("both a habit and a dealbreaker" in e for e in errors))


class ExtractionTests(unittest.TestCase):
    def test_valid_first_time_makes_one_call(self):
        api = FakeAPI(GOOD)
        self.assertEqual(pp.get_traits("some text", {}, call_fn=api)[0], GOOD)
        self.assertEqual(len(api.calls), 1)

    def test_invalid_then_valid_retries_once_with_the_errors(self):
        api = FakeAPI({**GOOD, "social_energy": "shy"}, GOOD)
        traits, _ = pp.get_traits("some text", {}, call_fn=api)
        self.assertEqual(traits, GOOD)
        self.assertEqual(len(api.calls), 2)
        self.assertIn("social_energy must be one of", api.calls[1][-1]["content"])

    def test_invalid_twice_raises(self):
        api = FakeAPI("nonsense", "still nonsense")
        with self.assertRaises(llm.LLMError):
            pp.get_traits("some text", {}, call_fn=api)

    def test_cache_prevents_a_second_call(self):
        cache, api = {}, FakeAPI(GOOD)
        pp.get_traits("same text", cache, call_fn=api)
        traits, status = pp.get_traits("same text", cache, call_fn=api)   # FakeAPI has no replies left
        self.assertEqual((traits, status), (GOOD, "cached"))

    def test_cache_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "cache.json")
            pp.get_traits("text", {}, cache_path=path, call_fn=FakeAPI(GOOD))
            _, status = pp.get_traits("text", llm.load_cache(path), call_fn=FakeAPI())
            self.assertEqual(status, "cached")

    def test_api_error_marks_student_failed_and_continues(self):
        students = [{"user_id": "u001", "first_name": "Mei", "email": "a@x.com",
                     "structured": {}, "text": "one"},
                    {"user_id": "u002", "first_name": "Tom", "email": "b@x.com",
                     "structured": {}, "text": "two"}]
        api = FakeAPI(llm.LLMError("HTTP 529"), GOOD)
        profiles, counts = pp.build_profiles(students, {}, call_fn=api)
        self.assertEqual([p["extraction_status"] for p in profiles], ["failed", "ok"])
        self.assertEqual(profiles[0]["traits"]["social_energy"], "unknown")

    def test_three_failures_in_a_row_stop_the_run(self):
        students = [{"user_id": f"u00{i}", "first_name": "X", "email": f"{i}@x.com", "structured": {}, "text": str(i)}
                    for i in range(5)]
        api = FakeAPI(*[llm.LLMError("network")] * 3)
        with self.assertRaises(llm.FatalLLMError):
            pp.build_profiles(students, {}, call_fn=api)
        self.assertEqual(len(api.calls), 3)

    def test_fatal_error_stops_the_run(self):
        students = [{"user_id": "u001", "first_name": "Mei", "email": "a@x.com", "structured": {}, "text": "x"}]
        with self.assertRaises(llm.FatalLLMError):
            pp.build_profiles(students, {}, call_fn=FakeAPI(llm.FatalLLMError("no key")))


class CsvCleaningTests(unittest.TestCase):
    def test_budget_range(self):
        self.assertEqual(pp.budget_range(["4000-4999", "5000-5999"]), (4000, 5999))
        self.assertEqual(pp.budget_range(["6000-7499", "7500+"]), (6000, None))
        self.assertEqual(pp.budget_range([]), (None, None))

    def test_labels_become_values(self):
        problems = []
        self.assertEqual(pp.option_value("smoker", "Only outdoors", problems), "outside")
        self.assertEqual(pp.option_value("campus", "HKMU", problems), "hkmu")                # "Other" box
        self.assertEqual(pp.option_value("dealbreakers", "Someone who’s usually up past 1am", problems),
                         "late_sleeper")                                                     # curly apostrophe
        self.assertEqual(problems, [])
        pp.option_value("smoker", "Sometimes", problems)
        self.assertEqual(len(problems), 1)

    def test_question_titles_map_to_columns(self):
        self.assertEqual(pp.canonical_header("How tidy do you keep shared spaces?"), "cleanliness")
        self.assertEqual(pp.canonical_header("cleanliness"), "cleanliness")

    def test_name_is_redacted(self):
        self.assertEqual(pp.redact_name("Hi, I'm Mei. MEI loves Meiji chocolate.", "Mei"),
                         "Hi, I'm [name]. [name] loves Meiji chocolate.")


if __name__ == "__main__":
    unittest.main()
