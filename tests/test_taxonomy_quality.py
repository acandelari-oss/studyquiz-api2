import unittest

import main


class TaxonomyQualityTests(unittest.TestCase):
    def _profiles(self, category_a, count_a, category_b, count_b):
        return {
            category_a: {"topic_count": count_a},
            category_b: {"topic_count": count_b},
        }

    def _metrics(
        self,
        category_a,
        category_b,
        lexical_similarity,
        centroid_similarity,
        cross_topic_affinity,
    ):
        return {
            "category_a": category_a,
            "category_b": category_b,
            "lexical_similarity": lexical_similarity,
            "centroid_similarity": centroid_similarity,
            "cross_topic_affinity": cross_topic_affinity,
        }

    def test_balanced_naming_variants_are_aliases(self):
        category_a = "DNA SEQUENCING TECHNIQUES"
        category_b = "GENETIC SEQUENCING TECHNOLOGIES"
        reason = main._is_category_alias(
            self._metrics(
                category_a,
                category_b,
                lexical_similarity=0.42,
                centroid_similarity=0.74,
                cross_topic_affinity=0.66,
            ),
            self._profiles(category_a, 2, category_b, 2),
        )

        self.assertEqual(
            reason,
            "balanced_lexical_semantic_alias",
        )

    def test_broad_and_narrow_categories_are_not_aliases(self):
        category_a = "INDAGINI PRELIMINARI"
        category_b = "CARATTERI DELLE INDAGINI PRELIMINARI"
        reason = main._is_category_alias(
            self._metrics(
                category_a,
                category_b,
                lexical_similarity=0.71,
                centroid_similarity=0.90,
                cross_topic_affinity=0.73,
            ),
            self._profiles(category_a, 9, category_b, 5),
        )

        self.assertIsNone(reason)

    def test_diagnostics_preserve_every_topic(self):
        ledger = (
            main.TaxonomyTopicLedgerEntry(
                "T0001",
                "CHROMOSOME ABNORMALITIES",
                "Numerical abnormalities",
                "Description",
                5,
                (0, 0),
                (1.0, 0.0),
            ),
            main.TaxonomyTopicLedgerEntry(
                "T0002",
                "CASES OF CHROMOSOMAL ABNORMALITIES",
                "Nondisjunction",
                "Description",
                5,
                (1, 0),
                (0.99, 0.01),
            ),
        )
        mapping = {
            "CHROMOSOME ABNORMALITIES": "CHROMOSOME ABNORMALITIES",
            "CASES OF CHROMOSOMAL ABNORMALITIES": (
                "CHROMOSOME ABNORMALITIES"
            ),
        }
        groups = [{
            "canonical_category": "CHROMOSOME ABNORMALITIES",
            "source_categories": [
                "CASES OF CHROMOSOMAL ABNORMALITIES",
                "CHROMOSOME ABNORMALITIES",
            ],
            "topic_count": 2,
        }]
        diagnostics = main.build_taxonomy_quality_diagnostics(
            topic_ledger=ledger,
            category_mapping=mapping,
            final_groups=groups,
            accepted_merges=[],
        )

        self.assertEqual(diagnostics["topic_count_before"], 2)
        self.assertEqual(diagnostics["topic_count_after"], 2)
        self.assertEqual(diagnostics["category_count_before"], 2)
        self.assertEqual(diagnostics["category_count_after"], 1)


if __name__ == "__main__":
    unittest.main()
