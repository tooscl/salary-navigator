import csv
import io
import json
import unittest

from utils.build_dataset import assemble, digest, split_csv


class DatasetPartsTests(unittest.TestCase):
    def test_round_trip_multiline_and_search_history(self):
        columns = ['source', 'id', 'description', 'search_filter_observations']
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows([
            dict(source='avito', id='1', description='one\n"two"; three',
                 search_filter_observations='[["run1","Москва",""],["run2",null,"Курьер"]]'),
            dict(source='avito', id='2', description='a,b', search_filter_observations='[]'),
        ])
        original = output.getvalue().encode('utf-8-sig')
        files, parts = split_csv(original)
        manifest = dict(columns=columns, rows=2, sha256=digest(original), parts=parts)
        self.assertEqual(assemble(manifest, files.__getitem__), original)
        self.assertEqual(split_csv(original), (files, parts))
        files[parts[0]['path']] += b'corruption'
        with self.assertRaises(ValueError):
            assemble(manifest, files.__getitem__)

    def test_duplicate_key_rejected(self):
        data = b'source,id\navito,1\navito,1\n'
        files, parts = split_csv(data)
        with self.assertRaisesRegex(ValueError, 'Duplicate key'):
            assemble(dict(columns=['source', 'id'], rows=2, sha256=digest(data), parts=parts), files.__getitem__)
