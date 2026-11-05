"""Mocked grouping contracts, not a claim of live-model semantic accuracy."""
import copy
import json
import unittest
from taxonomy_organization import organize_categories


class GroupingTests(unittest.TestCase):
    def setUp(self):
        subjects = [
            ("Definizione e principi del concorso di reati", "Pluralità di reati"),
            ("Cumulo materiale e giuridico", "Determinazione della pena nel concorso di reati"),
            ("Imputato", "Posizione processuale e diritti"),
            ("Diritti dell'imputato", "Garanzie difensive"),
            ("Struttura Organizzativa del Pubblico Ministero", "Organizzazione degli uffici requirenti"),
            ("Funzioni del Pubblico Ministero", "Esercizio dell'azione penale"),
            ("Ruolo della Polizia Giudiziaria", "Compiti investigativi"),
            ("Attività della Polizia Giudiziaria", "Attività di indagine"),
        ]
        self.topics = [dict(assignment_id=f"uuid-{i}", title=title, description=scope,
                            document_id="doc", evidence_ids=[i, i+100])
                       for i, (title, scope) in enumerate(subjects)]
        self.data = {"categories": [{"name": "Procedura Penale", "topics": self.topics}]}
        self.categories = [
            {"name": "Concorso di reati e cumulo delle pene", "description": "Pluralità di reati e trattamento sanzionatorio"},
            {"name": "Imputato e garanzie difensive", "description": "Posizione processuale e diritti dell'imputato"},
            {"name": "Pubblico Ministero", "description": "Uffici requirenti e azione penale"},
            {"name": "Polizia Giudiziaria", "description": "Organi e attività investigative"}]
        self.rows = [{"topic_id": f"T{i+1:04d}", "category_id": f"C{i//2:04d}"} for i in range(8)]
        self.calls = []

    def request(self, prompt, stage):
        self.calls.append((prompt, stage))
        return ({"categories": self.categories} if stage == "category_proposal" else
                {"assignments": self.rows, "quality_issues": []})

    def test_subject_groups_preserve_related_topics_and_distinguish_actors(self):
        before = copy.deepcopy(self.data)
        result = organize_categories(self.data, "infer", [], self.request)
        self.assertEqual([c["name"] for c in result["categories"]], [c["name"] for c in self.categories])
        self.assertEqual([[t["assignment_id"] for t in c["topics"]] for c in result["categories"]],
                         [[f"uuid-{i}", f"uuid-{i+1}"] for i in range(0, 8, 2)])
        self.assertEqual(self.data, before)
        self.assertTrue(all(t is self.topics[int(t["assignment_id"].split("-")[1])]
                            for c in result["categories"] for t in c["topics"]))
        self.assertEqual([stage for _, stage in self.calls], ["category_proposal", "category_assignment"])
        proposal = self.calls[0][0]
        self.assertIn("first identify distinct subjects and form coherent provisional groups", proposal)
        self.assertIn("Only after forming these groups", proposal)
        self.assertIn("Input list order and topic IDs are not evidence", proposal)
        self.assertNotIn("Procedura Penale", proposal)  # Previous label does not anchor proposal.
        for prompt, _ in self.calls:
            payload = json.loads(prompt.split("ORGANIZATION INPUT:\n")[1])
            self.assertEqual([t["title"] for t in payload["topics"]], [t["title"] for t in self.topics])
            self.assertLessEqual(len(prompt), 48000)

    def test_grouping_does_not_relax_coverage_or_unique_assignment(self):
        valid = copy.deepcopy(self.rows)
        for rows in (valid[:-1], valid + [valid[0]]):
            self.rows = rows
            self.calls.clear()
            before = copy.deepcopy(self.data)
            with self.assertRaises(ValueError):
                organize_categories(self.data, "infer", [], self.request)
            self.assertEqual(len(self.calls), 3 if len(rows) < len(valid) else 2)
            self.assertEqual(self.data, before)
