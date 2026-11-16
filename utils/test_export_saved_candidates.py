import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from utils.export_saved_candidates import archive_candidates, verify_parent_file


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

    def test_archive_keeps_raw_and_original_selection_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'archive'
            source = Path(directory) / 'source'
            batch = source / 'Data/avito_runs/parent'
            (batch / 'raw').mkdir(parents=True)
            (batch / 'exports').mkdir()
            raw = {'state': 'DETAIL', 'id': '123', 'title': 'Оператор',
                   'fields': [{'name': 'Формат работы', 'value': 'Гибрид'}]}
            payload = json.dumps(raw).encode()
            (batch / 'raw/123.json').write_bytes(payload)
            (batch / 'raw/456.json').write_text('{"state":"UNKNOWN_PAGE"}')
            (batch / 'exports/reviewed_selection.json').write_text(json.dumps({'decisions': [
                {'id': '123', 'accepted': False, 'reason': 'mixed format',
                 'raw_sha256': hashlib.sha256(payload).hexdigest()},
            ]}))
            summary = archive_candidates(source, ['parent'], 'archive', root=root)
            self.assertEqual(summary['observations'], 2)
            with (root / 'Data/avito_runs/archive/observations.csv').open() as stream:
                excluded, incomplete = list(csv.DictReader(stream))
            self.assertEqual(excluded['selection_status'], 'excluded')
            self.assertEqual(excluded['selection_reason'], 'mixed format')
            self.assertEqual((root / excluded['raw_path']).read_bytes(), payload)
            self.assertEqual(excluded['Параметр — Формат работы'], 'Гибрид')
            self.assertEqual(incomplete['selection_status'], 'incomplete_observation')
