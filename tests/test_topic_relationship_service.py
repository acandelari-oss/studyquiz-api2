import unittest

from topic_relationship_service import (
    TopicRelationshipFilters,
    build_topic_relationship_graph,
    get_topic_relationship_graph,
    semantic_similarity_from_negative_inner_product,
)


TOPICS = [
    {
        "id": "topic-a",
        "topic": "Batteri Lattici",
        "category": "Microbiologia alimentare",
        "source_section": "Fermentazione",
    },
    {
        "id": "topic-b",
        "topic": "Fermentazione Lattica",
        "category": "Biochimica",
        "source_section": "Metabolismo",
    },
    {
        "id": "topic-c",
        "topic": "Respirazione",
        "category": "Metabolismo",
        "source_section": "Respirazione",
    },
    {
        "id": "topic-d",
        "topic": "Topic Isolato",
        "category": "Altro",
        "source_section": "Altro",
    },
]


class TopicRelationshipServiceTests(unittest.TestCase):
    def _graph(
        self,
        topic_rows=None,
        topic_chunk_rows=None,
        semantic_pair_rows=None,
        filters=None,
    ):
        return build_topic_relationship_graph(
            topic_rows if topic_rows is not None else TOPICS,
            topic_chunk_rows if topic_chunk_rows is not None else [
                {
                    "topic_id": "topic-a",
                    "chunk_id": 1,
                    "document_id": "doc-1",
                    "doc_title": "Microbiology.pdf",
                    "section": "Fermentazione",
                    "text_length": 100,
                },
                {
                    "topic_id": "topic-b",
                    "chunk_id": 1,
                    "document_id": "doc-1",
                    "doc_title": "Microbiology.pdf",
                    "section": "Fermentazione",
                    "text_length": 100,
                },
                {
                    "topic_id": "topic-a",
                    "chunk_id": 2,
                    "document_id": "doc-1",
                    "doc_title": "Microbiology.pdf",
                    "section": "Fermentazione",
                    "text_length": 120,
                },
                {
                    "topic_id": "topic-b",
                    "chunk_id": 2,
                    "document_id": "doc-1",
                    "doc_title": "Microbiology.pdf",
                    "section": "Fermentazione",
                    "text_length": 120,
                },
                {
                    "topic_id": "topic-c",
                    "chunk_id": 3,
                    "document_id": "doc-2",
                    "doc_title": "Respiration.pdf",
                    "section": "Respirazione",
                    "text_length": 90,
                },
            ],
            semantic_pair_rows if semantic_pair_rows is not None else [
                {
                    "topic_a_id": "topic-a",
                    "topic_b_id": "topic-b",
                    "semantic_similarity": 0.72,
                    "negative_inner_product": None,
                },
                {
                    "topic_a_id": "topic-a",
                    "topic_b_id": "topic-c",
                    "semantic_similarity": 0.49,
                    "negative_inner_product": None,
                },
                {
                    "topic_a_id": "topic-b",
                    "topic_b_id": "topic-c",
                    "semantic_similarity": 0.51,
                    "negative_inner_product": None,
                },
            ],
            filters=filters or TopicRelationshipFilters(),
        )

    def test_every_display_topic_is_returned_as_node(self):
        graph = self._graph()

        self.assertEqual(graph["node_count"], 4)
        self.assertEqual(
            {node["id"] for node in graph["nodes"]},
            {"topic-a", "topic-b", "topic-c", "topic-d"},
        )

    def test_non_display_topics_are_excluded_by_loader_contract(self):
        graph = self._graph(topic_rows=TOPICS[:2])

        self.assertEqual(graph["node_count"], 2)
        self.assertEqual(
            {node["id"] for node in graph["nodes"]},
            {"topic-a", "topic-b"},
        )

    def test_cross_category_relationships_are_allowed(self):
        graph = self._graph()

        edge = graph["edges"][0]
        self.assertEqual(edge["topic_a_id"], "topic-a")
        self.assertEqual(edge["topic_b_id"], "topic-b")
        self.assertNotEqual(edge["category_a"], edge["category_b"])

    def test_category_does_not_influence_relationship_strength(self):
        changed_categories = [
            {**topic, "category": "Same Category"}
            for topic in TOPICS
        ]

        original = self._graph()
        changed = self._graph(topic_rows=changed_categories)

        self.assertEqual(
            original["edges"][0]["semantic_similarity"],
            changed["edges"][0]["semantic_similarity"],
        )
        self.assertEqual(
            original["edges"][0]["shared_chunks"],
            changed["edges"][0]["shared_chunks"],
        )
        self.assertEqual(
            original["edges"][0]["chunk_jaccard"],
            changed["edges"][0]["chunk_jaccard"],
        )

    def test_shared_chunks_and_jaccard_are_counted_correctly(self):
        graph = self._graph()
        edge = graph["edges"][0]

        self.assertEqual(edge["shared_chunks"], 2)
        self.assertEqual(edge["chunks_a"], 2)
        self.assertEqual(edge["chunks_b"], 2)
        self.assertEqual(edge["chunk_jaccard"], 1.0)
        self.assertEqual(edge["shared_sections"], 1)
        self.assertEqual(edge["shared_documents"], 1)

    def test_duplicate_topic_chunk_rows_do_not_inflate_counts(self):
        duplicate_links = [
            {
                "topic_id": "topic-a",
                "chunk_id": 1,
                "document_id": "doc-1",
                "doc_title": "Microbiology.pdf",
                "section": "Fermentazione",
                "text_length": 100,
            },
            {
                "topic_id": "topic-a",
                "chunk_id": 1,
                "document_id": "doc-1",
                "doc_title": "Microbiology.pdf",
                "section": "Fermentazione",
                "text_length": 100,
            },
            {
                "topic_id": "topic-b",
                "chunk_id": 1,
                "document_id": "doc-1",
                "doc_title": "Microbiology.pdf",
                "section": "Fermentazione",
                "text_length": 100,
            },
        ]

        graph = self._graph(topic_chunk_rows=duplicate_links)
        edge = graph["edges"][0]

        self.assertEqual(edge["shared_chunks"], 1)
        self.assertEqual(edge["chunks_a"], 1)
        self.assertEqual(edge["chunks_b"], 1)

    def test_semantic_similarity_sign_is_correct_for_negative_inner_product(self):
        self.assertEqual(
            semantic_similarity_from_negative_inner_product(-0.84),
            0.84,
        )

        graph = self._graph(
            topic_chunk_rows=[],
            semantic_pair_rows=[
                {
                    "topic_a_id": "topic-a",
                    "topic_b_id": "topic-c",
                    "semantic_similarity": None,
                    "negative_inner_product": -0.81,
                },
            ],
            filters=TopicRelationshipFilters(
                min_semantic_similarity=0.8,
                min_shared_chunks=1,
                top_k_per_topic=5,
            ),
        )

        self.assertEqual(graph["edges"][0]["semantic_similarity"], 0.81)

    def test_topic_with_no_qualifying_edges_remains_in_nodes(self):
        graph = self._graph()

        self.assertIn(
            "topic-d",
            {node["id"] for node in graph["nodes"]},
        )
        self.assertGreaterEqual(graph["isolated_node_count"], 1)

    def test_focus_topic_returns_strongest_relationships_for_selected_topic(self):
        graph = self._graph(
            filters=TopicRelationshipFilters(
                min_semantic_similarity=0.0,
                min_shared_chunks=1,
                top_k_per_topic=1,
                focus_topic="Batteri Lattici",
            ),
        )

        self.assertEqual(graph["edge_count"], 1)
        self.assertEqual(graph["filters"]["focus_topic_id"], "topic-a")
        self.assertTrue(
            graph["edges"][0]["topic_a_id"] == "topic-a"
            or graph["edges"][0]["topic_b_id"] == "topic-a"
        )

    def test_no_database_writes_are_used_by_loader(self):
        class FakeDb:
            def __init__(self):
                self.statements = []

            def execute(self, statement, params=None):
                self.statements.append(str(statement).lower())

                class Result:
                    def fetchall(self):
                        return []

                return Result()

        db = FakeDb()
        get_topic_relationship_graph(db, "project-1")

        combined_sql = "\n".join(db.statements)
        self.assertNotIn("insert ", combined_sql)
        self.assertNotIn("update ", combined_sql)
        self.assertNotIn("delete ", combined_sql)


if __name__ == "__main__":
    unittest.main()
