import itertools
import json
import unittest
from collections import Counter
from unittest.mock import patch

from quiz_options import shuffle_question_options


class QuizOptionTests(unittest.TestCase):
    def test_all_permutations_preserve_answer_and_cover_all_five_positions(self):
        for correct in range(5):
            positions = Counter()
            for permutation in itertools.permutations(range(5)):
                question = {"options": ["a", "b", "c", "d", "e"], "correct_answer": correct, "correct": correct, "explanation": "Reasoning"}
                correct_text = question["options"][correct]
                with patch("quiz_options.random.shuffle", side_effect=lambda order: order.__setitem__(slice(None), permutation)):
                    shuffle_question_options(question)
                self.assertEqual(question["options"][question["correct_answer"]], correct_text)
                self.assertEqual(question["correct"], question["correct_answer"])
                self.assertEqual(question["explanation"], "Reasoning")
                self.assertCountEqual(question["options"], ["a", "b", "c", "d", "e"])
                positions[question["correct_answer"]] += 1
            self.assertEqual(positions, dict.fromkeys(range(5), 24))

    def test_saved_options_and_index_survive_json_roundtrip(self):
        question = {"options": ["A", "B", "C", "D", "E"], "correct_answer": 0, "answer": 0}
        with patch("quiz_options.random.shuffle", side_effect=lambda order: order.reverse()):
            shuffle_question_options(question)
        loaded = json.loads(json.dumps(question))
        self.assertEqual(loaded["correct_answer"], 4)
        self.assertEqual(loaded["answer"], 4)
        self.assertEqual(loaded["options"][4], "A")

    def test_duplicate_labels_are_tracked_by_original_index(self):
        question = {"options": ["same", "same", "other"], "correct_answer": 0}
        with patch("quiz_options.random.shuffle", side_effect=lambda order: order.reverse()):
            shuffle_question_options(question)
        self.assertEqual(question["correct_answer"], 2)

    def test_invalid_indices_fail_before_mutation(self):
        for index in [-1, 5, None, "0", True]:
            question = {"options": ["a", "b", "c", "d", "e"], "correct_answer": index}
            before = dict(question)
            with self.assertRaises(ValueError):
                shuffle_question_options(question)
            self.assertEqual(question, before)
