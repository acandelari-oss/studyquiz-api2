import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import main


class AskAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.http = TestClient(main.app)
        self.db = MagicMock()
        self.db.execute.return_value.fetchone.return_value = ("project",)
        self.answer = {"answer": "Explanation", "sources": []}
        self.session_patch = patch.object(main, "SessionLocal", return_value=self.db)
        self.answer_patch = patch.object(main, "answer_ask_question", return_value=self.answer)
        self.session_patch.start()
        self.generate = self.answer_patch.start()
        main.app.dependency_overrides[main.verify_user] = lambda: {"id": "student"}

    def tearDown(self):
        main.app.dependency_overrides.clear()
        self.answer_patch.stop()
        self.session_patch.stop()
        self.http.close()

    def post(self, image=False):
        if image:
            return self.http.post("/ask_with_image", data={"project_id": "project", "question": "Why?"}, files={"image": ("test.png", b"image", "image/png")})
        return self.http.post("/ask", json={"project_id": "project", "question": "Why?", "history": [{"role": "user", "content": "Earlier"}], "expand_search": True})

    def test_missing_auth_never_reaches_retrieval(self):
        main.app.dependency_overrides.clear()
        for image in [False, True]:
            self.assertEqual(self.post(image).status_code, 401)
        self.generate.assert_not_called()

    def test_other_users_project_never_reaches_retrieval(self):
        self.db.execute.return_value.fetchone.return_value = None
        for image in [False, True]:
            self.assertEqual(self.post(image).status_code, 404)
        self.generate.assert_not_called()
        self.assertEqual(self.db.close.call_count, 2)

    def test_owned_project_preserves_history_and_search_mode(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.answer)
        params = self.db.execute.call_args.args[1]
        self.assertEqual(params, {"project_id": "project", "user_id": "student"})
        self.generate.assert_called_once_with(project_id="project", question="Why?", topics=[], history=[{"role": "user", "content": "Earlier"}], expand_search=True)
        self.db.close.assert_called_once()

    def test_owned_image_request_still_works(self):
        self.assertEqual(self.post(True).status_code, 200)
        self.assertTrue(self.generate.call_args.kwargs["image_data_url"].startswith("data:image/png;base64,"))
