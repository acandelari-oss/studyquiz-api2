import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException
import main


class QuestionAudioTests(unittest.TestCase):
    def setUp(self):
        self.http = TestClient(main.app)
        main.app.dependency_overrides[main.verify_user] = lambda: {"id": "user"}
        self.db = MagicMock()
        self.client = MagicMock()
        self.client.with_options.return_value.audio.speech.create.return_value.content = b"mp3-data"
        self.patches = [patch.object(main, "SessionLocal", return_value=self.db),
                        patch.object(main, "_require_owned_project"),
                        patch.object(main, "client", self.client)]
        self.mocks = [p.start() for p in self.patches]

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        main.app.dependency_overrides.clear()
        self.http.close()

    def post(self, question):
        return self.http.post("/projects/project/active_recall_question_audio", json={"question": question})

    def test_exact_question_and_fixed_voice(self):
        question = "  Perché le piante crescono?\nExplain photosynthesis."
        response = self.post(question)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"mp3-data")
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        self.assertEqual(response.headers["cache-control"], "no-store")
        args = self.client.with_options.return_value.audio.speech.create.call_args.kwargs
        self.assertEqual(args["input"], question)
        self.assertEqual(args["voice"], "marin")
        self.assertEqual(args["model"], "gpt-4o-mini-tts")
        self.mocks[1].assert_called_once_with(self.db, "project", "user")
        self.db.close.assert_called_once()

    def test_auth_required(self):
        main.app.dependency_overrides.clear()
        self.assertEqual(self.post("Question?").status_code, 401)
        self.client.with_options.assert_not_called()

    def test_ownership_required(self):
        self.mocks[1].side_effect = HTTPException(404, "Project not found")
        self.assertEqual(self.post("Question?").status_code, 404)
        self.client.with_options.assert_not_called()
        self.db.close.assert_called_once()

    def test_invalid_input_does_not_call_openai(self):
        for question in ["", "  \n", "x" * 4097]:
            self.assertEqual(self.post(question).status_code, 422)
        self.client.with_options.assert_not_called()

    def test_provider_failure_is_sanitized(self):
        self.client.with_options.return_value.audio.speech.create.side_effect = RuntimeError("private credentials")
        response = self.post("Question?")
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("private credentials", response.text)
