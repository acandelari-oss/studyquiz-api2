"""Randomize display positions once, immediately before quiz persistence."""
import random


def shuffle_question_options(question):
    options = question.get("options")
    correct_index = question.get("correct_answer")
    if not isinstance(options, list) or not options:
        raise ValueError("Question must have answer options")
    if type(correct_index) is not int or not 0 <= correct_index < len(options):
        raise ValueError("Question must have a valid normalized correct-answer index")

    # Track original indices rather than matching text, including duplicate labels.
    order = list(range(len(options)))
    random.shuffle(order)
    question["options"] = [options[index] for index in order]
    question["correct_answer"] = order.index(correct_index)
    if "correct" in question:
        question["correct"] = question["correct_answer"]
    if type(question.get("answer")) is int:
        question["answer"] = question["correct_answer"]
