import os
import tempfile
import unittest

import backend.storage as storage


class BackendStorageTests(unittest.TestCase):
    def test_web_index_state_is_mirrored_to_sqlite(self):
        original_dir = storage.WEB_INDEX_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                storage.WEB_INDEX_DIR = tmpdir
                path = os.path.join(tmpdir, "tasks.json")
                payload = [{"task_id": "1", "status": "done"}]
                storage.write_json(path, payload)
                os.remove(path)

                self.assertEqual(storage.read_json(path, []), payload)
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "state.sqlite3")))
            finally:
                storage.WEB_INDEX_DIR = original_dir


if __name__ == "__main__":
    unittest.main()
