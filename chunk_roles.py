import re
import unicodedata
from collections import Counter


CHUNK_ROLES = (
    "teaching",
    "exercise",
    "intro",
    "outline",
    "bibliography",
    "administrative",
    "cover",
)

def _normalize_role_text(value):
    value = str(value or "").replace("’", "'").replace("‘", "'")
    normalized = "".join(
        character
        for character in unicodedata.normalize(
            "NFKD",
            value.lower(),
        )
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _count_matches(text, patterns):
    return sum(1 for pattern in patterns if pattern in text)


def classify_chunk_role(
    chunk_text,
    page_number=None,
    doc_title=None,
):
    if any(_EXERCISE_HEADING.fullmatch(_normalize_role_text(line)) for line in str(chunk_text or "").splitlines()):
        return "exercise"
    text = _normalize_role_text(chunk_text)
    title = _normalize_role_text(doc_title)
    page = int(page_number) if page_number is not None else None
    early_page = page is not None and page <= 3

    cover_institution_patterns = (
        "university",
        "universita",
        "universite",
        "universidad",
        "hochschule",
        "faculty",
        "facolta",
        "department",
        "dipartimento",
        "institut",
        "school of",
    )
    cover_person_patterns = (
        "professor",
        "prof.",
        "prof ",
        "docente",
        "instructor",
        "lecturer",
        "teacher",
    )
    cover_contact_patterns = (
        "e-mail",
        "email",
        "@",
        "tel.",
        "telefono",
        "phone",
        "office",
        "ufficio",
        "stanza",
        "website",
        "www.",
        "http://",
        "https://",
    )

    institution_hits = _count_matches(text, cover_institution_patterns)
    person_hits = _count_matches(text, cover_person_patterns)
    contact_hits = _count_matches(text, cover_contact_patterns)
    title_repeated = bool(title and title in text)

    if early_page and (
        contact_hits >= 2
        or (
            institution_hits >= 1
            and contact_hits >= 1
            and (person_hits >= 1 or title_repeated)
        )
    ):
        return "cover"

    bibliography_patterns = (
        "bibliography",
        "bibliografia",
        "references",
        "riferimenti bibliografici",
        "recommended books",
        "recommended reading",
        "further reading",
        "textbooks",
        "testi consigliati",
        "libri consigliati",
        "letture consigliate",
        "ouvrages recommandes",
        "literaturverzeichnis",
        "empfohlene literatur",
        "referencias bibliograficas",
        "bibliografia recomendada",
    )
    publishing_patterns = (
        "oxford press",
        "cambridge press",
        "springer",
        "elsevier",
        "wiley",
        "zanichelli",
        "isbn",
        "edition",
        "edizione",
        "editore",
    )

    if (
        _count_matches(text, bibliography_patterns) >= 1
        or (
            early_page
            and _count_matches(text, publishing_patterns) >= 2
        )
    ):
        return "bibliography"

    administrative_patterns = (
        "exam information",
        "exam format",
        "assessment method",
        "grading",
        "grade distribution",
        "modalita di esame",
        "modalita d'esame",
        "prova scritta",
        "prova orale",
        "office hours",
        "orario di ricevimento",
        "course credits",
        "crediti formativi",
        "ects",
        "attendance",
        "frequenza obbligatoria",
        "course schedule",
        "lesson schedule",
        "calendario del corso",
        "calendario delle lezioni",
    )

    # Administrative labels must be complete words/phrases. Substring matching
    # made "ECTS" match teaching words such as protects, detects and defects.
    if any(re.search(r"(?<!\w)" + re.escape(pattern) + r"(?!\w)", text)
           for pattern in administrative_patterns):
        return "administrative"

    outline_patterns = (
        "course objectives",
        "learning objectives",
        "learning outcomes",
        "course outline",
        "course syllabus",
        "syllabus",
        "course contents",
        "contents of the course",
        "topics covered",
        "course structure",
        "obiettivi formativi",
        "obiettivi del corso",
        "risultati di apprendimento",
        "programma del corso",
        "contenuti del corso",
        "struttura del corso",
        "programme du cours",
        "objectifs du cours",
        "lernziele",
        "kursinhalte",
        "programa del curso",
        "objetivos del curso",
    )

    if _count_matches(text, outline_patterns) >= 1:
        return "outline"

    intro_patterns = (
        "introduction",
        "introduzione",
        "overview",
        "panoramica",
        "background",
        "why study",
        "perche studiare",
        "is a discipline",
        "e una disciplina",
        "si definisce come",
        "can be defined as",
        "in this chapter",
        "in questo capitolo",
        "this chapter introduces",
        "questo capitolo introduce",
        "motivational example",
        "esempio motivazionale",
        "wikipedia",
    )

    if _count_matches(text, intro_patterns) >= 1:
        return "intro"

    motivational_intro_patterns = (
        "why ",
        "perche ",
        "real-world application",
        "real world application",
        "applicazione",
        "applications",
        "one of the most",
        "uno dei",
        "is the key to",
        "is the secret",
        "e il segreto",
        "conversion of energy",
        "conversione dell'energia",
    )

    if (
        page is not None
        and page <= 10
        and len(text) <= 400
        and _count_matches(text, motivational_intro_patterns) >= 1
    ):
        return "intro"

    return "teaching"


def normalize_chunk_role(
    stored_role,
    chunk_text,
    page_number=None,
    doc_title=None,
):
    if stored_role in CHUNK_ROLES:
        return stored_role

    return classify_chunk_role(
        chunk_text,
        page_number=page_number,
        doc_title=doc_title,
    )


def is_assignment_eligible_chunk_role(role):
    return role == "teaching"


def count_chunk_roles(roles):
    counts = Counter(roles)
    return {
        role: counts.get(role, 0)
        for role in CHUNK_ROLES
    }


def log_chunk_role_counts(roles):
    counts = count_chunk_roles(roles)
    print("CHUNK ROLE COUNTS:", counts)
    print("COVER CHUNKS:", counts["cover"])
    print("OUTLINE CHUNKS:", counts["outline"])
    print("BIBLIOGRAPHY CHUNKS:", counts["bibliography"])
    print("ADMINISTRATIVE CHUNKS:", counts["administrative"])
    print("INTRO CHUNKS:", counts["intro"])
    print("TEACHING CHUNKS:", counts["teaching"])

    return counts

# Section boundaries are recognized before whitespace flattening and chunk overlap.
_EXERCISE_HEADING = re.compile(
    r"^(?:\d+[.)]\s*)?(?:esercizi|esercitazioni|exercises|practice questions|"
    r"review questions|self[- ]assessment|quiz|test di autovalutazione|"
    r"exercices|ubungen|ejercicios)(?:\s+(?:unita|unit|capitolo|chapter|"
    r"di riepilogo|di verifica|svolti|proposti)\b[^.!?]*)?\s*[:.]?$", re.I)
_TEACHING_HEADING = re.compile(
    r"^(?:chapter|capitolo|unit|unita|part|parte|lesson|lezione)\s+"
    r"(?:\d+|[ivxlcdm]+|[a-z])(?:\s*[-–—:.]\s*|\s+).+", re.I)


def _teaching_identity(text):
    match = re.search(r"\b(chapter|capitolo|unit|unita|part|parte|lesson|lezione)\s+(\d+|[ivxlcdm]+|[a-z])\b", text, re.I)
    if not match:
        return None
    names = {'capitolo':'chapter', 'unita':'unit', 'parte':'part', 'lezione':'lesson'}
    return (names.get(match[1].lower(), match[1].lower()), match[2].lower())


class ExerciseSectionFilter:
    """Document-local, deterministic exercise boundaries; no inferred answers.

    Exclusion continues until an unseen explicit chapter/unit/lesson heading.
    Repeated running headings do not end an exercise section. Unknown boundaries
    stay excluded rather than presenting possible distractors as factual evidence.
    """
    def __init__(self):
        self.in_exercises = False
        self.seen_teaching_headings = set()

    def split(self, text):
        result, buffered = [], []
        active = self.in_exercises

        def flush():
            if buffered:
                result.append(("".join(buffered), "exercise" if active else None))
                buffered.clear()

        lines = (text or "").splitlines(keepends=True)
        # A numbered question plus ordered answer labels is stronger evidence than
        # numbering alone. This also catches explicit MCQs without a section title.
        question_starts = set()
        question_start, options = None, []
        for index, line in enumerate(lines):
            value = line.strip()
            if re.match(r"^\d+[.)]\s+", value):
                question_start, options = index, []
            option = re.match(r"^([A-E])[.)]\s+", value)
            if question_start is not None and option:
                options.append(option.group(1))
                stem = " ".join(lines[question_start:index])
                if options == ['A', 'B', 'C'] and (
                    '?' in stem or re.search(r"\b(?:choose|select|scegli|seleziona)\b", stem, re.I)
                ):
                    question_starts.add(question_start)
        for index, line in enumerate(lines):
            normalized = _normalize_role_text(line)
            if index in question_starts or _EXERCISE_HEADING.fullmatch(normalized):
                flush()
                active = True
                identity = _teaching_identity(normalized)
                if identity:
                    self.seen_teaching_headings.add(identity)
            elif _TEACHING_HEADING.fullmatch(normalized):
                # Page numbers in a running header are not a new instructional unit.
                identity = _teaching_identity(normalized)
                if identity not in self.seen_teaching_headings:
                    if active:
                        flush()
                    active = False
                    self.seen_teaching_headings.add(identity)
            buffered.append(line)
        flush()
        self.in_exercises = active
        return result


def chunk_document_block(text, section_filter, clean, chunk):
    """Return unchanged text transforms, separated at exercise boundaries.

    The boolean signals whether ordinal-based, precomputed alignment is reusable.
    """
    original = [clean(c) for c in chunk(clean(text))]
    original = [c for c in original if c]
    spans = section_filter.split(text)
    result = [(value, role) for span, role in spans
              for c in chunk(clean(span)) if (value := clean(c))]
    return result, [value for value, _ in result] == original


def require_exercise_role_support(db):
    """Fail before acquisition/AI if the deployed PostgreSQL constraint is older."""
    if db.get_bind().dialect.name != 'postgresql':
        return
    from sqlalchemy import text
    definitions = db.execute(text("""
        SELECT pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = 'public.chunks'::regclass
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%chunk_role%'
    """)).scalars().all()
    if any("'exercise'" not in definition for definition in definitions):
        raise ValueError(
            "Exercise filtering requires migration 20261102_exercise_chunk_role.sql "
            "before uploading. No document processing has started.")
