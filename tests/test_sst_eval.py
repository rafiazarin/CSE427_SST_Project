"""Unit tests for the shared sst_eval helpers.

Run with:  python -m unittest discover -s tests   (no extra deps required)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sst_eval as S  # noqa: E402


class TestNormalize(unittest.TestCase):
    def test_strips_boilerplate_and_punct(self):
        self.assertEqual(S.normalize_answer("Final answer: Red."), "red")
        self.assertEqual(S.normalize_answer("It is a CAR"), "car")

    def test_number_words(self):
        self.assertEqual(S.normalize_answer("three"), "3")

    def test_articles_dropped(self):
        self.assertEqual(S.normalize_answer("the dog"), "dog")

    def test_unknown_markers(self):
        self.assertEqual(S.normalize_answer("I cannot determine"), "unknown")

    def test_missing(self):
        self.assertEqual(S.normalize_answer(None), "")
        self.assertEqual(S.normalize_answer(float("nan")), "")


class TestLenient(unittest.TestCase):
    def test_exact_and_substring(self):
        self.assertEqual(S.is_lenient_correct("red", "the red bus"), 1)
        self.assertEqual(S.is_lenient_correct("yes", "Yes"), 1)
        self.assertEqual(S.is_lenient_correct("red", "blue"), 0)

    def test_yesno_strict(self):
        # yes/no must match exactly, not be a substring of a sentence
        self.assertEqual(S.is_lenient_correct("no", "yes definitely"), 0)

    def test_unknown_pred_is_wrong(self):
        self.assertEqual(S.is_lenient_correct("red", "unknown"), 0)


class TestTokens(unittest.TestCase):
    def test_count_positive(self):
        self.assertGreater(S.count_tokens("a b c d"), 0)


class TestSST(unittest.TestCase):
    def setUp(self):
        self.scene = {
            "objects": {
                "1": {"name": "bus", "attributes": ["red"],
                      "relations": [{"name": "near", "object": "2"}]},
                "2": {"name": "person", "attributes": [], "relations": []},
                "3": {"name": "nose", "attributes": []},  # low-value -> dropped
            }
        }

    def test_low_value_filtered_and_relations_built(self):
        sst = S.gqa_scene_to_clean_sst(self.scene)
        self.assertIn("bus", sst["objects"])
        self.assertNotIn("nose", sst["objects"])
        self.assertIn("bus near person", sst["relations"])
        self.assertEqual(sst["text"], [])  # GQA scene graph has no OCR

    def test_field_ablations(self):
        sst = {"objects": ["bus"], "counts": ["bus: 2"], "attributes": ["bus: red"],
               "relations": ["bus near person"], "text": ["STOP"]}
        self.assertEqual(S.make_no_attributes_sst(sst)["attributes"], [])
        self.assertEqual(S.make_no_relations_sst(sst)["relations"], [])
        oo = S.make_objects_only_sst(sst)
        self.assertEqual(oo["objects"], ["bus"])
        self.assertEqual(oo["attributes"], [])


class TestKeywordFilter(unittest.TestCase):
    def test_keeps_question_relevant_objects(self):
        sample = {
            "question": "What color is the bus?",
            "sst": {"objects": ["bus", "tree", "sky"], "counts": [],
                    "attributes": ["bus: red", "tree: green"],
                    "relations": ["bus near tree"], "text": []},
        }
        out = S.keyword_aware_sst_filter(sample)
        self.assertIn("bus", out["objects"])
        # the bus's attribute survives; an unrelated sky has nothing kept
        self.assertTrue(any("bus" in a for a in out["attributes"]))


class TestPrompts(unittest.TestCase):
    def test_all_methods_build(self):
        sample = {
            "question": "What color is the bus?",
            "sst": {"objects": ["bus"], "counts": [], "attributes": ["bus: red"],
                    "relations": [], "text": []},
        }
        self.assertEqual(len(S.METHODS), 8)
        for m in S.METHODS:
            prompt, sp = S.build_method_prompt(sample, m)
            self.assertTrue(prompt)
        # question_only must not leak scene content (the "red" attribute is
        # scene info; "bus" appears in the question itself so isn't a leak).
        q_prompt, q_sp = S.build_method_prompt(sample, "question_only")
        self.assertNotIn("red", q_prompt)
        self.assertEqual(q_sp, "no scene information")

    def test_method_names_cover_methods(self):
        for m in S.METHODS:
            self.assertIn(m, S.METHOD_NAMES)


class TestStats(unittest.TestCase):
    def test_mcnemar_symmetric_balanced(self):
        # equal discordant counts -> p == 1.0
        self.assertEqual(S.exact_mcnemar_p(5, 5), 1.0)

    def test_mcnemar_extreme(self):
        self.assertLess(S.exact_mcnemar_p(20, 0), 0.001)

    def test_mcnemar_from_pairs(self):
        ref = [1, 1, 0, 0]
        meth = [1, 0, 1, 0]
        res = S.mcnemar_from_pairs(ref, meth)
        self.assertEqual(res["method_wins"], 1)
        self.assertEqual(res["reference_wins"], 1)
        self.assertEqual(res["discordant"], 2)

    def test_bootstrap_in_range(self):
        lo, hi = S.bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 0, 1, 1], n_boot=500)
        self.assertTrue(0.0 <= lo <= hi <= 1.0)


if __name__ == "__main__":
    unittest.main()
