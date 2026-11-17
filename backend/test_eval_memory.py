import unittest

from eval_memory import (
    QuestionResult,
    context_token_estimate,
    load_fixtures,
    rank_of_target,
    score_answer,
    summarize,
)

MARKERS = ["not mention", "don't know"]


class EvalMemoryTests(unittest.TestCase):
    def test_fixtures_are_consistent(self) -> None:
        fixtures = load_fixtures()
        keys = [turn["key"] for turn in fixtures["turns"]]
        transcripts = [turn["transcript"] for turn in fixtures["turns"]]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(transcripts), len(set(transcripts)))
        recent_keys = set(keys[-4:])
        for question in fixtures["questions"]:
            for target in question["targets"]:
                self.assertIn(target, keys)
                # Facts must be outside the recent window so search is exercised.
                self.assertNotIn(target, recent_keys)
            if question["kind"] == "negative":
                self.assertEqual(question["targets"], [])
            else:
                self.assertTrue(question["expect_any"])

    def test_rank_of_target_is_one_based(self) -> None:
        ranked = [{"transcript": "a"}, {"transcript": "b"}, {"transcript": "c"}]
        self.assertEqual(rank_of_target(ranked, {"b"}), 2)
        self.assertIsNone(rank_of_target(ranked, {"z"}))

    def test_scores_positive_answers_by_expected_terms(self) -> None:
        question = {"kind": "exact", "expect_any": ["bulls"]}
        self.assertTrue(score_answer("You said the Chicago Bulls.", question, MARKERS, []))
        self.assertFalse(score_answer("I'm not sure.", question, MARKERS, []))

    def test_negative_answers_must_admit_and_not_invent(self) -> None:
        question = {"kind": "negative", "expect_any": []}
        self.assertTrue(
            score_answer("You haven’t... you did not mention it.", question, MARKERS, [])
        )
        self.assertFalse(score_answer("Your brother is Priya.", question, MARKERS, ["priya"]))
        self.assertFalse(
            score_answer("I don't know, maybe Priya?", question, MARKERS, ["priya"])
        )

    def test_summary_rates(self) -> None:
        results = [
            QuestionResult("a", "exact", "q", search_rank=1, context_hit=True, context_tokens=10),
            QuestionResult("b", "paraphrase", "q", search_rank=4, context_hit=False, context_tokens=30),
            QuestionResult("c", "negative", "q", noise_at_3=2, context_tokens=20),
        ]
        summary = summarize(results)
        self.assertEqual(summary["all"]["recall@1"], 0.5)
        self.assertEqual(summary["all"]["recall@5"], 1.0)
        self.assertEqual(summary["all"]["mrr"], round((1 + 0.25) / 2, 3))
        self.assertEqual(summary["paraphrase"]["context_hit"], 0.0)
        self.assertEqual(summary["negative_avg_noise@3"], 2)
        self.assertEqual(summary["max_context_tokens"], 30)

    def test_context_token_estimate(self) -> None:
        context = [{"transcript": "x" * 40, "response": "y" * 80}]
        self.assertEqual(context_token_estimate(context), 30)


if __name__ == "__main__":
    unittest.main()
