"""Tests for matcher.py.   Run all tests from the project folder:  python -m unittest -v"""
import copy
import json
import os
import unittest

import config
import matcher
import parse_profiles


def make_profile(user_id="u001", **changes):
    """A neutral student. Pass structured fields directly, traits as traits={...}."""
    profile = {
        "user_id": user_id,
        "first_name": user_id,
        "structured": {
            "campus": "hku", "max_commute_min": 30, "preferred_areas": ["western"],
            "budget_min_hkd_month": 5000, "budget_max_hkd_month": 5999, "move_in_month": "2026-12",
            "sleep_schedule": "middle", "cleanliness": 3, "guests_frequency": 2, "noise_tolerance": 3,
            "smoker": "no", "has_pet": "no", "cooking": "sometimes",
            "languages": ["english"], "dealbreakers": [],
        },
        "traits": {"social_energy": "ambivert", "study_habits": "home_flexible",
                   "cooking_frequency": "sometimes", "weekend_lifestyle": "balanced",
                   "habits": [], "extra_dealbreakers": []},
    }
    profile["traits"].update(changes.pop("traits", {}))
    profile["structured"].update(changes)
    return profile


class HardFilterTests(unittest.TestCase):
    def s(self, **changes):
        return make_profile(**changes)["structured"]

    def test_budgets_that_overlap_pass(self):
        self.assertTrue(matcher.budgets_overlap(self.s(), self.s(budget_min_hkd_month=5500, budget_max_hkd_month=7499)))

    def test_neighbouring_bands_pass_thanks_to_tolerance(self):
        a = self.s(budget_min_hkd_month=4000, budget_max_hkd_month=4999)
        b = self.s(budget_min_hkd_month=5000, budget_max_hkd_month=5999)
        self.assertTrue(matcher.budgets_overlap(a, b))

    def test_distant_budgets_fail(self):
        a = self.s(budget_min_hkd_month=4000, budget_max_hkd_month=4999)
        b = self.s(budget_min_hkd_month=6000, budget_max_hkd_month=7499)
        self.assertFalse(matcher.budgets_overlap(a, b))
        self.assertFalse(matcher.budgets_overlap(b, a))       # symmetric

    def test_open_ended_budget(self):
        rich = self.s(budget_min_hkd_month=7500, budget_max_hkd_month=None)
        self.assertTrue(matcher.budgets_overlap(rich, self.s(budget_min_hkd_month=9000, budget_max_hkd_month=None)))
        self.assertFalse(matcher.budgets_overlap(rich, self.s(budget_min_hkd_month=0, budget_max_hkd_month=3999)))

    def test_move_in_adjacent_month_passes_two_months_fails(self):
        self.assertTrue(matcher.move_in_compatible(self.s(move_in_month="2026-11"), self.s(move_in_month="2026-12")))
        self.assertFalse(matcher.move_in_compatible(self.s(move_in_month="2026-11"), self.s(move_in_month="2027-01")))

    def test_flexible_matches_any_month(self):
        self.assertTrue(matcher.move_in_compatible(self.s(move_in_month="flexible"), self.s(move_in_month="2027-03")))

    def test_smoking_outdoors_triggers_smoker_dealbreaker(self):
        self.assertTrue(matcher.violates_dealbreakers(self.s(dealbreakers=["smoker"]), self.s(smoker="outside")))

    def test_dealbreakers_are_checked_both_ways(self):
        picky, pet_owner = self.s(dealbreakers=["pets"]), self.s(has_pet="yes")
        self.assertFalse(matcher.passes_hard_filters(picky, pet_owner))
        self.assertFalse(matcher.passes_hard_filters(pet_owner, picky))

    def test_each_structured_dealbreaker(self):
        cases = {"late_sleeper": {"sleep_schedule": "late"}, "frequent_guests": {"guests_frequency": 4},
                 "messy": {"cleanliness": 2}}
        for dealbreaker, other in cases.items():
            with self.subTest(dealbreaker):
                self.assertTrue(matcher.violates_dealbreakers(self.s(dealbreakers=[dealbreaker]), self.s(**other)))
        self.assertFalse(matcher.violates_dealbreakers(self.s(dealbreakers=["messy"]), self.s(cleanliness=3)))


class ScoringTests(unittest.TestCase):
    def test_identical_students_score_100(self):
        a, b = make_profile("u001"), make_profile("u002")
        for mode in ("ai", "baseline"):
            self.assertAlmostEqual(matcher.directional_score(a, b, mode)[0], 100)

    def test_harmonic_mean_punishes_one_sided_matches(self):
        self.assertAlmostEqual(matcher.harmonic_mean(90, 30), 45)
        self.assertLess(matcher.harmonic_mean(90, 30), (90 + 30) / 2)
        self.assertEqual(matcher.harmonic_mean(0, 0), 0)

    def test_cleanliness_is_directional(self):
        tidy, messy = make_profile("u001", cleanliness=5), make_profile("u002", cleanliness=2)
        tidy_view, _ = matcher.directional_score(tidy, messy, "baseline")
        messy_view, _ = matcher.directional_score(messy, tidy, "baseline")
        self.assertLess(tidy_view, messy_view)    # the tidy person minds; the messy one doesn't

    def test_noise_tolerant_person_doesnt_mind_guests(self):
        tolerant = make_profile("u001", noise_tolerance=5, guests_frequency=1)
        party = make_profile("u002", guests_frequency=5)
        parts = matcher.component_scores(tolerant, party, "baseline")
        self.assertEqual(parts["guests"], 1.0)

    def test_baseline_ignores_llm_traits(self):
        a = make_profile("u001")
        b1 = make_profile("u002", traits={"social_energy": "extrovert"})
        b2 = make_profile("u002", traits={"social_energy": "ambivert"})
        self.assertEqual(matcher.directional_score(a, b1, "baseline"), matcher.directional_score(a, b2, "baseline"))
        self.assertNotEqual(matcher.directional_score(a, b1, "ai")[0], matcher.directional_score(a, b2, "ai")[0])

    def test_unknown_traits_are_skipped_not_punished(self):
        a = make_profile("u001", traits={"social_energy": "unknown"})
        score, used = matcher.directional_score(a, make_profile("u002"), "ai")
        self.assertNotIn("social_energy", used)
        self.assertAlmostEqual(score, 100)

    def test_extra_dealbreaker_clash(self):
        a = make_profile("u001", traits={"extra_dealbreakers": ["parties"]})
        b = make_profile("u002", traits={"habits": ["parties"]})
        self.assertEqual(matcher.component_scores(a, b, "ai")["extra_dealbreakers"], 0.0)
        self.assertIsNone(matcher.component_scores(b, a, "ai")["extra_dealbreakers"])   # b has none

    def test_scores_stay_between_0_and_100(self):
        a = make_profile("u001", cleanliness=5, noise_tolerance=1, sleep_schedule="early",
                         traits={"study_habits": "home_needs_quiet", "extra_dealbreakers": ["parties"]})
        b = make_profile("u002", cleanliness=1, guests_frequency=5, sleep_schedule="late",
                         preferred_areas=["west"], campus="mq", cooking="most_days", languages=["korean"],
                         traits={"social_energy": "extrovert", "weekend_lifestyle": "out_a_lot", "habits": ["parties"]})
        score, _ = matcher.directional_score(a, b, "ai")
        self.assertGreaterEqual(score, 0)
        self.assertLess(score, 20)   # a terrible match scores low


class GraphAndHeapTests(unittest.TestCase):
    def test_no_edge_when_filter_fails_and_edges_are_mutual(self):
        a, b = make_profile("u001"), make_profile("u002")
        c = make_profile("u003", smoker="yes")
        a["structured"]["dealbreakers"] = ["smoker"]
        graph = matcher.build_graph([a, b, c], "ai")
        self.assertNotIn("u003", graph["u001"])
        self.assertEqual(graph["u001"]["u002"]["score"], graph["u002"]["u001"]["score"])

    def test_top_k_returns_best_k_in_order(self):
        neighbours = {f"u{i}": {"score": s} for i, s in enumerate([50, 90, 70, 10, 80])}
        order = {f"u{i}": i for i in range(5)}
        self.assertEqual(matcher.top_k(neighbours, 3, order), ["u1", "u4", "u2"])

    def test_top_k_ties_go_to_earlier_row(self):
        neighbours = {"u9": {"score": 80}, "u2": {"score": 80}, "u5": {"score": 80}}
        order = {"u2": 2, "u5": 5, "u9": 9}
        self.assertEqual(matcher.top_k(neighbours, 2, order), ["u2", "u5"])

    def test_top_k_with_fewer_neighbours_than_k(self):
        self.assertEqual(matcher.top_k({"u1": {"score": 5}}, 3, {"u1": 0}), ["u1"])


class PlantedPairsIntegrationTest(unittest.TestCase):
    """Runs the real pipeline on the fake students, using their known traits."""

    @classmethod
    def setUpClass(cls):
        paths = config.paths(sample=True)
        if not os.path.exists(paths["responses"]):
            raise unittest.SkipTest("run generate_test_data.py first")
        with open(paths["ground_truth"], encoding="utf-8") as f:
            truth = json.load(f)
        students, _ = parse_profiles.read_students(paths["responses"])
        profiles, _ = parse_profiles.build_profiles(students, {}, known_traits=truth["students"])
        cls.planted = truth["planted_pairs"]
        cls.graphs = {mode: matcher.build_graph(profiles, mode) for mode in ("ai", "baseline")}

    def test_planted_pairs(self):
        for pair in self.planted:
            a, b, expect = pair["a"], pair["b"], pair["expect"]
            with self.subTest(pair["scenario"]):
                if expect == "no_edge":
                    self.assertNotIn(b, self.graphs["ai"][a])
                elif expect == "best_match_both_modes":
                    for graph in self.graphs.values():
                        best = max(edge["score"] for edge in graph[a].values())
                        self.assertEqual(graph[a][b]["score"], best)
                elif expect == "baseline_best_but_ai_lower":
                    base, ai = self.graphs["baseline"][a], self.graphs["ai"][a]
                    self.assertEqual(base[b]["score"], max(e["score"] for e in base.values()))
                    self.assertLess(ai[b]["score"], max(e["score"] for e in ai.values()))


if __name__ == "__main__":
    unittest.main()
