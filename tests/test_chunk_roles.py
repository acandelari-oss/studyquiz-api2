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

    def test_administrative_substrings_do_not_exclude_teaching(self):
        passages = [
            'Shelterin protects telomeres from being recognized as DNA damage.',
            'Mismatch repair detects errors after DNA synthesis.',
            'MutL activity directs incision of the newly synthesized strand.',
            'Defects in DNA repair increase mutation accumulation.',
            'The electric field affects the motion of charged particles.',
            'The contract protects the rights of both parties.',
            'The policy reflects changes in demand and their economic effects.',
            'The enzyme acts by degrading the substrate.',
            'The process intersects with several other pathways.',
            '15. Final high-yield comparison tables: MSH detects mismatches; '
            'DNA glycosylase recognizes damaged bases; ligase seals the nick.',
            '16.4 If asked: Compare NHEJ and HR. Both repair double-strand '
            'breaks. Defects in HR proteins increase cancer risk.',
        ]
        for passage in passages:
            with self.subTest(passage=passage):
                role = classify_chunk_role(passage, page_number=10)
                self.assertEqual(role, 'teaching')
                self.assertTrue(is_assignment_eligible_chunk_role(role))

    def test_complete_administrative_words_and_phrases_still_excluded(self):
        passages = [
            'Course workload: 6 ECTS.',
            'Credits (ECTS): 6',
            'ECTS-based credit allocation.',
            'Exam information:\nAssessment method: written test.',
            'OFFICE\tHOURS: Monday, 10–12.',
            'Modalità d’esame: prova orale.',
            'Frequenza obbligatoria e crediti formativi.',
            'Grading: assignments 30%, final exam 70%.',
        ]
        for passage in passages:
            with self.subTest(passage=passage):
                role = classify_chunk_role(passage, page_number=10)
                self.assertEqual(role, 'administrative')
                self.assertFalse(is_assignment_eligible_chunk_role(role))

    def test_administrative_label_still_applies_with_incidental_teaching_words(self):
        self.assertEqual(classify_chunk_role(
            'Exam information: the test covers how DNA repair detects defects.',
            page_number=10), 'administrative')

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
