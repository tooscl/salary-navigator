"""Reproducible exact-name exclusion against a committed dataset snapshot."""
import csv
from collections import Counter
import hashlib
import io
import json
import subprocess

from utils.vacancy_labels import vacancy_labels, DEPENDENCIES

ROOT = __import__("pathlib").Path(__file__).resolve().parents[4]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def baseline(root, revision, city, profession):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    commit = git('rev-parse', '--verify', revision + '^{commit}').decode().strip()
    def blob(path):
        return git('show', f'{commit}:{path}')
    manifest_bytes = blob('Data/dataset/manifest.json')
    manifest = json.loads(manifest_bytes)
    expected_labels = manifest['vacancy_labels']['sha256']
    if {p.name: digest(p.read_bytes()) for p in DEPENDENCIES} != expected_labels:
        raise ValueError('Current label rules differ from baseline dataset')
    counts, ids, files = Counter(), set(), {}
    for part in manifest['parts']:
        name = part['path']
        if not name.startswith('parts/') or '..' in name.split('/'):
            raise ValueError('Invalid baseline part path')
        path = 'Data/dataset/' + name
        data = blob(path)
        if digest(data) != part['sha256']:
            raise ValueError('Baseline part hash differs from manifest')
        files[path] = digest(data)
        rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline='')))
        if len(rows) != part['rows']:
            raise ValueError('Baseline part row count differs')
        for row in rows:
            if row['id'] in ids: raise ValueError('Duplicate baseline ID')
            ids.add(row['id'])
            if row['city_resolved'] == city and row['profession_resolved'] == profession:
                counts[row['employer_name_observed']] += 1
    if not counts: raise ValueError('Empty baseline cohort')
    named = [(name, count) for name, count in counts.items() if name.strip()]
    if not named: raise ValueError('No named baseline employer')
    top = max(count for _, count in named)
    leaders = [name for name, count in named if count == top]
    if len(leaders) != 1: raise ValueError('Baseline employer leader is tied')
    return dict(commit=commit, manifest_sha256=digest(manifest_bytes), parts=files,
                city=city, profession=profession, cohort_count=sum(counts.values()),
                leader=leaders[0], leader_count=top, employer_counts=dict(sorted(counts.items())))


def select(rows, target, evidence, raw_hashes):
    if target <= 0 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Expected positive target and unique candidate IDs')
    decisions, accepted = [], []
    for row in rows:
        identity = row['id']
        if identity not in raw_hashes: raise ValueError('Missing candidate raw hash')
        labels = vacancy_labels(row)
        name = row['employer_name_observed']
        if labels['city_resolved'] != evidence['city']: reason = 'city_mismatch_or_unknown'
        elif labels['profession_resolved'] != evidence['profession']: reason = 'profession_mismatch_or_unknown'
        elif not name.strip(): reason = 'employer_unknown'
        elif name == evidence['leader']: reason = 'excluded_leader_exact_match'
        elif len(accepted) >= target: reason = 'after_target'
        else:
            reason = 'accepted'; accepted.append(identity)
        decisions.append(dict(id=identity, raw_sha256=raw_hashes[identity],
                              employer_name_observed=name, **labels, decision=reason))
    if len(accepted) != target: raise ValueError('Employer selection target not reached')
    return dict(method='exact_observed_employer_exclusion_v1', baseline=evidence, target=target,
                label_dependencies={p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in DEPENDENCIES},
                decisions=decisions, accepted_ids=accepted)
