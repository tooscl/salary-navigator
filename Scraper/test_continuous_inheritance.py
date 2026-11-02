import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent/'skills/collect-avito/scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('continuous', SCRIPTS/'continuous.py')
continuous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuous)


class InheritanceTests(unittest.TestCase):
    def test_only_completed_pinned_matching_bytes_are_inherited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root/'code.js'; code.write_text('verified', encoding='utf8')
            digest = hashlib.sha256(code.read_bytes()).hexdigest()
            prior = root/'Scraper/runs/prior'; source = root/'Scraper/runs/source'
            continuous.write(prior/'plan.json', {'automation_sha256': {'code.js':digest}})
            continuous.write(source/'plan.json', {'automation_sha256': {'code.js':digest}})
            manifest = {'status':'complete', 'files': {'Scraper/runs/prior/plan.json': {'sha256':hashlib.sha256((prior/'plan.json').read_bytes()).hexdigest()}},
                        'automation_changes':[{'file':'code.js','reason':'fix','validation':'test passed'}]}
            continuous.write(prior/'manifest.json', manifest)
            completed = [{'capture_run':'prior'}]
            continuous.inherit_validated_changes(root, 'source', completed)
            path = source/'diagnostics/automation_changes.json'
            record = continuous.read(path)[0]
            self.assertEqual(record['validated_file_sha256'], digest)
            self.assertEqual(record['validation_source_manifest_sha256'], hashlib.sha256((prior/'manifest.json').read_bytes()).hexdigest())
            path.unlink(); code.write_text('changed', encoding='utf8')
            continuous.inherit_validated_changes(root, 'source', completed)
            self.assertFalse(path.exists())
            code.write_text('verified', encoding='utf8'); manifest['status']='incomplete'
            continuous.write(prior/'manifest.json', manifest)
            continuous.inherit_validated_changes(root, 'source', completed)
            self.assertFalse(path.exists())


if __name__ == '__main__': unittest.main()
