import unittest

from chunk_roles import (
    classify_chunk_role,
    is_assignment_eligible_chunk_role,
)


class ChunkRoleClassificationTests(unittest.TestCase):
    def test_cover_page(self):
        role = classify_chunk_role(
            (
                "Università di Padova Dipartimento di Chimica "
                "Professor Rossi email: rossi@example.it "
                "tel. 049 123456 website www.example.it"
            ),
            page_number=1,
            doc_title="Chimica Organica",
        )
        self.assertEqual(role, "cover")

    def test_outline(self):
        role = classify_chunk_role(
            "Obiettivi formativi e struttura del corso.",
            page_number=2,
        )
        self.assertEqual(role, "outline")

    def test_bibliography_precedes_exam_information(self):
        role = classify_chunk_role(
            (
                "Testi consigliati e recommended books. "
                "Modalità di esame: prova scritta."
            ),
            page_number=3,
        )
        self.assertEqual(role, "bibliography")

    def test_administrative(self):
        role = classify_chunk_role(
            "Exam information, grading, office hours and course schedule.",
            page_number=3,
        )
        self.assertEqual(role, "administrative")

    def test_intro(self):
        role = classify_chunk_role(
            "Introduction: organic chemistry is a discipline within chemistry.",
            page_number=4,
        )
        self.assertEqual(role, "intro")

    def test_early_motivational_example_is_intro(self):
        role = classify_chunk_role(
            (
                "A molecular machine is one of the most striking "
                "applications of organic chemistry."
            ),
            page_number=6,
        )
        self.assertEqual(role, "intro")

    def test_teaching_default(self):
        role = classify_chunk_role(
            "A nucleophile donates an electron pair to an electrophile.",
            page_number=12,
        )
        self.assertEqual(role, "teaching")

    def test_only_teaching_role_is_eligible(self):
        self.assertTrue(is_assignment_eligible_chunk_role("teaching"))

        for role in (
            "intro",
            "outline",
            "cover",
            "bibliography",
            "administrative",
        ):
            self.assertFalse(is_assignment_eligible_chunk_role(role))


if __name__ == "__main__":
    unittest.main()
