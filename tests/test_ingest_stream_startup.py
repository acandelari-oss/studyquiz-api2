import unittest
from unittest.mock import MagicMock, patch

from fastapi import BackgroundTasks

import main


class IngestStreamStartupTests(unittest.IsolatedAsyncioTestCase):
    async def check_startup(self, existing_name=None, continuation=False):
        db = MagicMock()
        db.execute.return_value.scalar.return_value = 1
        db.execute.return_value.fetchone.return_value = (
            "module-id", existing_name, 1, False
        )
        request = main.IngestRequest(
            documents=[main.IngestDocument(title="notes.pdf", file_bytes="dGVzdA==")],
            module_name="  Biology  ",
            module_id="module-id" if continuation else None,
        )
        with patch.object(main, "SessionLocal", return_value=db), \
             patch.object(main, "_require_owned_project"), \
             patch.object(main, "_update_study_module_organization_metadata"), \
             patch.object(main, "UploadPipelineLogger"):
            response = await main.ingest_stream(
                "project-id", request, BackgroundTasks(), user={"id": "user-id"}
            )
            try:
                first_message = await response.body_iterator.__anext__()
                expected_name = existing_name or "Biology"
                self.assertEqual(
                    first_message, f"Starting upload for module: {expected_name}\n"
                )
                inserts = [
                    call for call in db.execute.call_args_list
                    if "insert into study_modules" in str(call.args[0])
                ]
                self.assertEqual(len(inserts), 0 if continuation else 1)
                if inserts:
                    self.assertEqual(inserts[0].args[1]["name"], "Biology")
                db.commit.assert_called_once()
            finally:
                await response.body_iterator.aclose()
        db.close.assert_called_once()

    async def test_new_module_starts_upload_with_requested_name(self):
        await self.check_startup()

    async def test_continuation_preserves_existing_module_name(self):
        await self.check_startup("Existing biology", continuation=True)

    async def test_continuation_empty_name_uses_requested_name(self):
        await self.check_startup("", continuation=True)
