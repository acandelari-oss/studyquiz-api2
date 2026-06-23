LANGUAGE_REGISTRY = {
    "it": {
        "code": "it",
        "name": "Italian",
        "native_name": "Italiano",
        "enabled": True,
    },
    "en": {
        "code": "en",
        "name": "English",
        "native_name": "English",
        "enabled": True,
    },
}


def normalize_bcp47_tag(value):
    if not value:
        return ""

    parts = str(value).strip().replace("_", "-").split("-")
    parts = [part for part in parts if part]

    if not parts:
        return ""

    normalized = [parts[0].lower()]

    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            normalized.append(part.title())
        elif (
            len(part) == 2 and part.isalpha()
        ) or (
            len(part) == 3 and part.isdigit()
        ):
            normalized.append(part.upper())
        else:
            normalized.append(part.lower())

    return "-".join(normalized)


def get_enabled_language(value):
    normalized = normalize_bcp47_tag(value)
    language = LANGUAGE_REGISTRY.get(normalized)

    if not language or not language.get("enabled"):
        return None

    return language


def get_enabled_languages():
    return [
        language
        for language in LANGUAGE_REGISTRY.values()
        if language.get("enabled")
    ]


def get_enabled_language_codes():
    return [
        language["code"]
        for language in get_enabled_languages()
    ]
