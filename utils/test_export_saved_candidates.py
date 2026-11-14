import hashlib
import tempfile
import unittest
from pathlib import Path

from utils.export_saved_candidates import verify_parent_file


class ParentProvenanceTests(unittest.TestCase):
    def test_rejects_changed_or_unregistered_parent_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'raw.json'
            path.write_bytes(b'original')
            manifest = {'files': {'raw.json': {'sha256': hashlib.sha256(b'original').hexdigest()}}}
            verify_parent_file(root, manifest, path)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_parent_file(root, manifest, path)
            with self.assertRaises(ValueError):
                verify_parent_file(root, {'files': {}}, path)
