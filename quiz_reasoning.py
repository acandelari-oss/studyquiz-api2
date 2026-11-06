"""Local evidence-contract checks for generated reasoning questions.

These checks validate explicit evidence and the declared task, not arbitrary
semantic entailment. No model call or subject-specific knowledge is used.
"""
import re
import unicodedata

OPERATIONS = {"application", "comparison", "distinction", "consequence", "exception", "synthesis"}


def _text(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip() if isinstance(value, str) else ""


def evaluate_reasoning_evidence(question, material):
    reasons = []
    if not _text(question.get("question")):
        return {"valid": False, "reasons": ["invalid_question_stem"]}
    options = question.get("options")
    if (not isinstance(options, list) or len(options) != 5
            or any(not _text(option) for option in options)
            or len({_text(option).casefold() for option in options}) != 5):
        return {"valid": False, "reasons": ["invalid_answer_options"]}
    answer = question.get("correct_answer")
    if answer is None:
        answer = question.get("correct", question.get("answer"))
    index = None
    if isinstance(answer, int) and not isinstance(answer, bool):
        index = answer
    elif isinstance(answer, str):
        answer = answer.strip()
        if answer.isdigit():
            index = int(answer)
        elif answer.upper() in list("ABCDE"):
            index = ord(answer.upper()) - ord("A")
        elif answer in options:
            index = options.index(answer)
    if index is None or not 0 <= index < 5:
        return {"valid": False, "reasons": ["invalid_correct_answer"]}
    reasoning = question.get("reasoning")
    if not isinstance(reasoning, dict):
        return {"valid": False, "reasons": ["missing_reasoning_evidence"]}
    if reasoning.get("operation") not in OPERATIONS:
        reasons.append("not_a_reasoning_task")
    by_id = {str(c["chunk_id"]): _text(c.get("chunk_text")) for c in material}
    citations = question.get("source_chunk_ids")
    if not isinstance(citations, list) or not citations:
        reasons.append("invalid_question_sources")
        cited = set()
    else:
        cited = {str(value) for value in citations if isinstance(value, (str, int)) and not isinstance(value, bool)}
        if (any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in citations)
                or len(cited) == 0 or not cited <= set(by_id)):
            reasons.append("invalid_question_sources")
    premises = reasoning.get("premises")
    unique_quotes = set()
    if not isinstance(premises, list):
        premises = []
    for premise in premises:
        if not isinstance(premise, dict):
            reasons.append("invalid_reasoning_premise")
            continue
        chunk_id = str(premise.get("chunk_id", ""))
        quote = _text(premise.get("quote"))
        if chunk_id not in by_id or chunk_id not in cited:
            reasons.append("reasoning_source_outside_scope")
        elif not quote or quote not in by_id[chunk_id]:
            reasons.append("reasoning_quote_not_in_source")
        else:
            unique_quotes.add(quote)
    if len(unique_quotes) < 2:
        reasons.append("insufficient_distinct_premises")
    derivation = _text(reasoning.get("derivation"))
    if not derivation or derivation == "...":
        reasons.append("missing_reasoning_derivation")
    elif derivation in unique_quotes:
        reasons.append("derivation_only_repeats_premise")
    if _text(reasoning.get("conclusion")) != _text(options[index]):
        reasons.append("reasoning_answer_mismatch")
    if not _text(question.get("explanation")):
        reasons.append("missing_reasoning_explanation")
    return {"valid": not reasons, "reasons": list(dict.fromkeys(reasons))}


def reasoning_evidence_instructions():
    return '''
REASONING EVIDENCE CONTRACT:
Include a "reasoning" object in each question, using this structure:
{"operation":"application", "premises":[{"chunk_id":"source ID", "quote":"exact source excerpt"},
{"chunk_id":"source ID", "quote":"a different exact source excerpt"}],
"derivation":"Explain how the premises together justify the selected answer",
"conclusion":"Exact text of the correct option"}.
Allowed operations: application, comparison, distinction, consequence, exception, synthesis.
The student must apply or relate at least two distinct source-supported premises.
A definition, isolated mechanism lookup, or recognition of a recalled statement
is not reasoning. Context words and a long stem do not change that.
For direct recall use operation "recall"; do not mislabel it to pass validation.
Premise quotes must be copied verbatim from the supplied chunk_text, with the
corresponding chunk_id also included in source_chunk_ids. Two copies of the
same fact do not establish two premises. Never invent a quote or premise.
The derivation must connect the premises to the answer, not just restate a fact.
The explanation shown to the student must explain the same reasoning.
Use any clear question opening; no narrative framing or minimum stem length is required.
Return an empty questions array if no supported reasoning question can be formed.
'''
