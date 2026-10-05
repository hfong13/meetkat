"""Tests for build_experiment.py and analyze.py."""
import json
import os
import unittest

import analyze
import build_experiment as be
import config
from tests.test_matcher import make_profile


def ranked(*ids):
    return [{"user_id": i} for i in ids]


class ChooseThreeTests(unittest.TestCase):
    def test_disagreement_picks_skip_the_other_lists_top_3(self):
        ai = ranked("a", "b", "c", "d", "e")
        base = ranked("b", "a", "x", "c", "y")
        picks, design = be.choose_three(ai, base)
        self.assertEqual(design, "disagreement")
        # a, b are in baseline's top 3 -> AI's first picks it "owns" are c (#3) and d (#4)
        # x is baseline's #3 and not in AI's top 3 -> baseline's pick
        self.assertEqual(picks, [("c", "ai", 3), ("d", "ai", 4), ("x", "baseline", 3)])

    def test_both_sides_follow_the_same_rule(self):
        picks, _ = be.choose_three(ranked("a", "b", "c", "d", "e"), ranked("v", "w", "x"))
        self.assertEqual(picks, [("a", "ai", 1), ("b", "ai", 2), ("v", "baseline", 1)])

    def test_baseline_pick_never_repeats_an_ai_pick(self):
        picks, _ = be.choose_three(ranked("a", "b", "c", "d", "e"), ranked("a", "b", "c", "d", "e"))
        self.assertEqual(len({uid for uid, _, _ in picks}), 3)

    def test_fallback_when_lists_are_identical(self):
        picks, design = be.choose_three(ranked("a", "b", "c", "d"), ranked("a", "b", "c", "d"))
        self.assertEqual(design, "fallback")
        self.assertEqual(picks, [("a", "ai", 1), ("b", "ai", 2), ("c", "baseline", 3)])

    def test_not_enough_matches(self):
        self.assertEqual(be.choose_three(ranked("a"), ranked("a", "b")), (None, None))
        self.assertEqual(be.choose_three(ranked("a", "b"), ranked("a", "b")), (None, None))


class PresentationTests(unittest.TestCase):
    def test_chips_are_capped_and_fallback_exists(self):
        a, b = make_profile("u001", sleep_schedule="early", cleanliness=5), make_profile("u002", sleep_schedule="early", cleanliness=5)
        chips = be.shared_traits(a, b)
        self.assertEqual(len(chips), 3)
        self.assertEqual(chips[0], "Both early sleepers")
        c = make_profile("u003", sleep_schedule="late", cleanliness=2, campus="mq", preferred_areas=["west"],
                         traits={"social_energy": "extrovert", "weekend_lifestyle": "out_a_lot"})
        self.assertEqual(be.shared_traits(a, c), ["Budgets and move-in dates line up"])

    def test_explanation_validator(self):
        validate = be.make_validator("Mei")
        self.assertEqual(validate("You and Mei both keep early hours.")[1], [])
        self.assertTrue(validate("Our AI thinks Mei fits.")[1])
        self.assertTrue(validate("One. Two. Three.")[1])
        self.assertTrue(validate("You both like quiet evenings.")[1])     # doesn't name Mei

    def test_app_data_never_contains_algorithm_labels(self):
        path = config.paths(sample=True)["matches"]
        if not os.path.exists(path):
            self.skipTest("run the sample pipeline first")
        with open(path, encoding="utf-8") as f:
            text = f.read().lower()
        for word in ("baseline", '"ai"', "source", "rank"):
            self.assertNotIn(word, text)


class AnalysisTests(unittest.TestCase):
    def test_sign_flip_exact_small_case(self):
        # 3 students, all +1. Of 8 sign patterns only all-plus and all-minus
        # have |mean| >= 1, so p = 2/8.
        self.assertAlmostEqual(analyze.sign_flip_test([1, 1, 1]), 0.25)

    def test_no_difference_gives_p_1(self):
        self.assertEqual(analyze.sign_flip_test([0, 0, 0, 0]), 1.0)

    def test_paired_analysis(self):
        joined = []
        for student, (ai1, ai2, base) in {"s1": (5, 4, 3), "s2": (4, 4, 4), "s3": (5, 5, 2)}.items():
            joined += [{"rater": student, "source": "ai", "rank": 1, "rating": ai1, "message": True},
                       {"rater": student, "source": "ai", "rank": 2, "rating": ai2, "message": False},
                       {"rater": student, "source": "baseline", "rank": 1, "rating": base, "message": False}]
        result, pairs = analyze.analyze(joined)
        self.assertEqual(result["top_ai_match"]["rated_4_or_5"], 3)
        self.assertAlmostEqual(result["by_source"]["baseline"]["mean_rating"], 3)
        self.assertAlmostEqual(result["by_source"]["ai"]["pct_would_message"], 50)
        # primary: best-ranked AI pick (rank 1) vs baseline -> +2, 0, +3
        self.assertAlmostEqual(result["paired"]["mean_difference"], 5 / 3)
        self.assertEqual((result["paired"]["ai_higher"], result["paired"]["tied"]), (2, 1))
        # secondary: average of both AI picks vs baseline -> +1.5, 0, +3
        self.assertAlmostEqual(result["paired"]["secondary_both_ai_picks"]["mean_difference"], 1.5)

    def test_top_ai_match_is_the_best_ranked_ai_pick_not_rank_1(self):
        joined = [{"rater": "s1", "source": "ai", "rank": 4, "rating": 2, "message": False},
                  {"rater": "s1", "source": "ai", "rank": 3, "rating": 5, "message": True},
                  {"rater": "s1", "source": "baseline", "rank": 3, "rating": 3, "message": False}]
        result, pairs = analyze.analyze(joined)
        self.assertEqual(result["top_ai_match"]["rated_4_or_5"], 1)
        self.assertEqual(pairs["s1"], (5, 3))


if __name__ == "__main__":
    unittest.main()
