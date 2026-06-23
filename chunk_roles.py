import re
import unicodedata
from collections import Counter


CHUNK_ROLES = (
    "teaching",
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

    if _count_matches(text, administrative_patterns) >= 1:
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
