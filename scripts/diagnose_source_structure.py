#!/usr/bin/env python3
import argparse
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from document_extractors import extract_uploaded_document
from source_structure import (
    analyze_source_structure,
    format_source_structure_tree,
    source_structure_to_json,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only diagnostic for DOUNO source-structure detection. "
            "It extracts a local document and prints detected headings before "
            "taxonomy generation uses them."
        )
    )
    parser.add_argument("path", help="Local PDF, DOCX, or PPTX file to inspect")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of the human-readable tree",
    )
    args = parser.parse_args()

    filename = os.path.basename(args.path)
    with open(args.path, "rb") as document_file:
        file_bytes = document_file.read()

    extracted_document = extract_uploaded_document(file_bytes, filename)
    analysis = analyze_source_structure(extracted_document)

    if args.json:
        print(source_structure_to_json(analysis))
    else:
        print(format_source_structure_tree(analysis))


if __name__ == "__main__":
    main()
