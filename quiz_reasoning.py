"""Local evidence-contract checks for generated reasoning questions.

These checks validate explicit evidence and the declared task, not arbitrary
semantic entailment. No model call or subject-specific knowledge is used.
"""
import re
import unicodedata

OPERATIONS = {"application", "comparison", "distinction", "consequence", "exception", "synthesis"}


def _text(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip() if isinstance(value, str) else ""


def _source_text(value):
    # Expand typographic ligatures only. NFKC would also flatten meaningful
    # mathematical notation (e.g. superscripts); global case folding changes
    # case-sensitive symbols and must not be used for scientific evidence.
    return _text(value).translate(str.maketrans({
        "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
        "ﬅ": "st", "ﬆ": "st",
    }))


def _verified_excerpt(quote, source):
    """Return the matched source excerpt, never a paraphrase or another source."""
    quote, source = _source_text(quote), _source_text(source)
    if not quote:
        return None
    if quote in source:
        return quote
    # A copied passage can be made into a sentence by the generator. Ignore
    # one terminal sentence mark, but never interior punctuation or characters.
    excerpt = quote[:-1] if quote.endswith((".", "!", "?")) else quote
    if not excerpt:
        return None
    variants = [excerpt]
    # Only ordinary multi-letter opening words; preserve symbols such as X/x.
    if re.match(r"^[A-Z][a-z]{2,}\b", excerpt):
        variants.append(excerpt[0].lower() + excerpt[1:])
    for variant in variants:
        for match in re.finditer(re.escape(variant), source):
            start, end = match.span()
            if ((start == 0 or not source[start - 1].isalnum())
                    and (end == len(source) or not source[end].isalnum())):
                return source[start:end]
    return None


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
        elif not quote or _verified_excerpt(quote, by_id[chunk_id]) is None:
            reasons.append("reasoning_quote_not_in_source")
        else:
            unique_quotes.add(_verified_excerpt(quote, by_id[chunk_id]))
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


def evidence_catalog(material):
    """Request-local, verbatim passages; IDs never refer outside selected sources."""
    catalog = {}
    for chunk in material:
        source = chunk.get('chunk_text') or ''
        # Preserve exact characters, including scientific notation. Splitting is
        # only for addressability; it does not certify a passage as a fact.
        for passage in re.split(r'(?<=[.!?])\s+|\n+', source):
            passage = passage.strip()
            if passage:
                catalog[f'E{len(catalog) + 1}'] = {
                    'chunk_id': str(chunk['chunk_id']), 'quote': passage}
    return catalog


def resolve_evidence_references(question, catalog):
    """Expand only server-owned IDs, then let the existing validator decide."""
    from copy import deepcopy
    result = deepcopy(question)
    reasoning = result.get('reasoning')
    if not isinstance(reasoning, dict):
        return result
    premises = reasoning.get('premises')
    if not isinstance(premises, list):
        return result
    resolved = []
    for premise in premises:
        if not isinstance(premise, dict) or set(premise) != {'evidence_id'}:
            resolved.append({})
            continue
        resolved.append(dict(catalog.get(premise['evidence_id'], {})))
    reasoning['premises'] = resolved
    # Citations are derived from the selected passages, never guessed by AI.
    result['source_chunk_ids'] = list(dict.fromkeys(
        p['chunk_id'] for p in resolved if 'chunk_id' in p))
    return result


def reasoning_response_format():
    string = {'type': 'string'}
    def obj(properties):
        return {'type': 'object', 'properties': properties,
                'required': list(properties), 'additionalProperties': False}
    question = obj({
        'question': string, 'options': {'type': 'array', 'items': string},
        'correct': {'type': 'integer'}, 'topic': string,
        'explanation': string, 'explanation_long': string,
        'source_document': string, 'source_page': {'type': ['integer', 'null']},
        'reasoning': obj({
            'operation': {'type': 'string', 'enum': sorted(OPERATIONS | {'recall'})},
            'premises': {'type': 'array', 'items': obj({'evidence_id': string})},
            'derivation': string, 'conclusion': string})})
    return {'type': 'json_schema', 'json_schema': {
        'name': 'reasoning_questions', 'strict': True,
        'schema': obj({'questions': {'type': 'array', 'items': question}})}}


def reference_evidence_instructions(catalog):
    import json
    return '''REASONING EVIDENCE CONTRACT:
Return questions matching the supplied JSON schema, including reasoning.
Select at least two distinct passages from EVIDENCE CATALOG by evidence_id.
Do not copy, rewrite or invent quotes or source IDs. The server resolves IDs.
Use two premises that together support an inference, not two repetitions of a fact.
A passage ID establishes provenance only, not correctness of an inference.
The derivation must explain how the premises justify the answer; conclusion must
exactly equal the correct option. A recalled statement is operation recall.
Do not label recall as reasoning. Return questions: [] if evidence is insufficient.
EVIDENCE CATALOG:
''' + json.dumps(catalog, ensure_ascii=False)
