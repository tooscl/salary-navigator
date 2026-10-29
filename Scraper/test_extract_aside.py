"""Explicit browser regression against localhost fixtures only: python3 Scraper/test_extract_aside.py."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import importlib.util
from pathlib import Path
import subprocess
import threading
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def fixture_runner():
    spec = importlib.util.spec_from_file_location('fixture_runner', ROOT / 'Scraper/skills/collect-avito/scripts/run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


class FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        html = (ROOT / 'Scraper/tests/fixtures/detail_123.html').read_text()
        if 'expanded' in self.path:
            html = html.replace('<div>График:', '<button>Показать ещё</button><div>График:')
        if 'duplicate' in self.path:
            html = html.replace('Опыт работы: Без опыта', 'График: Вахта')
        if 'unparsed' in self.path:
            html = html.replace('Опыт работы: Без опыта', 'Опыт работы<br>Без опыта')
        if 'blocked' in self.path:
            html += '<div role="dialog">Доступ ограничен: проблема с IP</div>'
        if self.path.startswith('/listing-'):
            checked = 'checked' if self.path in ('/listing-selected', '/listing-nested-selected') else ''
            role = '' if self.path == '/listing-nested-selected' else 'role="checkbox" aria-checked="' + ('true' if checked else 'false') + '"'
            control = '' if self.path == '/listing-missing' else (
                '<label ' + role + ' data-marker="params[144170]/checkbox/3201765">Курьер'
                '<input type="checkbox" data-marker="other-marker" value="3201765" ' + checked + '></label>')
            pager = ''
            if self.path.startswith('/listing-pager-'):
                scenario = self.path.removeprefix('/listing-pager-')
                next_button = ('<li class="listItem_arrow_next-test"><button aria-disabled="true"><svg data-icon-name="' + ('arrowleft' if scenario == 'wrong-direction' else 'arrowright') + '"></svg></button></li>' if scenario in ('disabled-next', 'wrong-direction') else '')
                if scenario == 'active-next':
                    next_button = '<li class="listItem_arrow_next-test"><a aria-label="Следующая страница" href="?p=25"><svg data-icon-name="arrowright"></svg></a></li>'
                pager = ('<ul data-marker="pagination-button"><li class="listItem_arrow_prev-test"><button aria-disabled="true"><svg data-icon-name="arrowleft"></svg></button></li><li><span aria-disabled="true">•••</span></li><li><span aria-current="page">24</span></li>' + next_button + '</ul>')
            html = ('<html><body><h1>Работа и вакансии</h1>' + control + pager +
                '<label data-marker="params[999]/checkbox/1"><input type="checkbox" checked>Гибкий график</label>'
                '<div data-marker="item"><a data-marker="item-title" href="/detail_123">Вакансия</a></div>'
                '</body></html>')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, *args):
        pass


class ExtractorBrowserTests(unittest.TestCase):
    def test_reusable_batch_saves_raw_and_stops_before_next_card(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        runner = fixture_runner()
        origin = f'http://127.0.0.1:{server.server_port}'
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); diag = root / 'diagnostics'; diag.mkdir()
                queue = [dict(id='123', url=origin+'/ready/detail_123', listing_rank=1),
                         dict(id='456', url=origin+'/blocked/detail_456', listing_rank=2),
                         dict(id='789', url=origin+'/ready/detail_789', listing_rank=3)]
                cp, elapsed = runner.browser(dict(mode='capture', queue=queue, completed_ids=[]), root/'out', diag)
                self.assertEqual(cp['state'], 'NEEDS_HUMAN')
                self.assertEqual(cp['completed_ids'], ['123'])
                self.assertEqual(cp['next_id'], '456')
                self.assertTrue((root/'out/raw/123.json').exists())
                self.assertTrue((root/'out/raw/456.json').exists())
                self.assertFalse((root/'out/raw/789.json').exists())
                self.assertTrue((diag/'problem.snapshot.txt').exists())
                self.assertGreater(elapsed, 0)
                print('Reusable batch: first raw saved, block checkpoint preserved, third URL not opened')
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_listing_profession_audit_selected_unfiltered_and_unreadable(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        runner = fixture_runner()
        script = """
var extractor = __EXTRACTOR__;
var results = [];
var p;
try {
  for (var test of ['selected', 'nested-selected', 'unfiltered', 'missing']) {
    if (!p) p = await openTab(__ORIGIN__ + '/listing-' + test);
    else await p.goto(__ORIGIN__ + '/listing-' + test);
    await snapshot(p);
    results.push({test, data: await p.evaluate(extractor)});
  }
} finally { if (p) await closeTab(p); }
console.log('TL_LISTINGS=' + JSON.stringify(results));
""".replace('__EXTRACTOR__', json.dumps((ROOT/'Scraper/extract.js').read_text()))
        script = script.replace('__ORIGIN__', json.dumps(f'http://127.0.0.1:{server.server_port}'))
        try:
            result = subprocess.run(runner.aside_command(runner.find_aside()) + ['repl', script],
                                    capture_output=True, text=True, timeout=90)
            lines = [line[len('TL_LISTINGS='):] for line in result.stdout.splitlines()
                     if line.startswith('TL_LISTINGS=')]
            self.assertEqual(result.returncode, 0)
            self.assertTrue(lines, 'Local listing fixture returned no results')
            selected, nested, unfiltered, missing = [row['data'] for row in json.loads(lines[0])]
            self.assertEqual(runner.profession_from_listing(selected), 'Курьер')
            self.assertEqual(runner.profession_from_listing(nested), 'Курьер')
            self.assertEqual(nested['profession_filter'], {'observed': True, 'selected_count': 1})
            self.assertEqual(selected['profession_filter'], {'observed': True, 'selected_count': 2})
            self.assertEqual(runner.profession_from_listing(unfiltered), '')
            self.assertEqual(unfiltered['profession_filter'], {'observed': True, 'selected_count': 0})
            self.assertFalse(missing['profession_filter']['observed'])
            with self.assertRaises(ValueError):
                runner.profession_from_listing(missing)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_listing_disabled_next_is_scoped_and_directional(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        runner = fixture_runner()
        script = """
var extractor = __EXTRACTOR__;
var results = [];
var p;
try {
  for (var scenario of ['previous-ellipsis', 'wrong-direction', 'active-next', 'disabled-next']) {
    if (!p) p = await openTab(__ORIGIN__ + '/listing-pager-' + scenario);
    else await p.goto(__ORIGIN__ + '/listing-pager-' + scenario);
    await snapshot(p);
    results.push(await p.evaluate(extractor));
  }
} finally { if (p) await closeTab(p); }
console.log('TL_PAGINATION=' + JSON.stringify(results));
""".replace('__EXTRACTOR__', json.dumps((ROOT/'Scraper/extract.js').read_text()))
        script = script.replace('__ORIGIN__', json.dumps(f'http://127.0.0.1:{server.server_port}'))
        try:
            result = subprocess.run(runner.aside_command(runner.find_aside()) + ['repl', script], capture_output=True, text=True, timeout=90)
            output = result.stdout + result.stderr
            lines = [line[len('TL_PAGINATION='):] for line in output.splitlines() if line.startswith('TL_PAGINATION=')]
            self.assertEqual(result.returncode, 0, output)
            self.assertTrue(lines, output)
            previous, wrong_direction, active, next_page = [x['pagination'] for x in json.loads(lines[0])]
            self.assertFalse(previous['end_confirmed'])
            self.assertIsNone(previous['evidence'])
            self.assertFalse(wrong_direction['end_confirmed'])
            self.assertFalse(active['end_confirmed'])
            self.assertTrue(active['next_url'].endswith('?p=25'))
            self.assertTrue(next_page['end_confirmed'])
            self.assertEqual(next_page['evidence']['direction'], 'next_arrowright')
            self.assertIsNone(next_page['next_url'])
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_visible_location_expansion_duplicates_and_blocking(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        script = '''
const extractor = __EXTRACTOR__;
const results = [];
let p;
try {
  for (const test of ['ready', 'expanded', 'duplicate', 'unparsed', 'blocked']) {
    const url = __ORIGIN__ + '/' + test + '/detail_123';
    if (!p) p = await openTab(url); else await p.goto(url);
    const snap = await snapshot(p);
    const data = await p.evaluate(extractor);
    let ok;
    if (test === 'blocked') ok = data.state === 'NEEDS_HUMAN' && !data.fields;
    else if (test === 'ready') ok = data.id === '123' && data.location_raw[0] === 'Москва, пос. Коммунарка' && data.parameter_audit.rendered_complete && data.fields.length === 2;
    else ok = !data.parameter_audit.rendered_complete;
    if (!ok) console.log(snap.tree);
    results.push({test, ok});
  }
} finally { if (p) await closeTab(p); }
console.log('TL_TESTS=' + JSON.stringify(results));
'''.replace('__EXTRACTOR__', json.dumps((ROOT/'Scraper/extract.js').read_text()))
        script = script.replace('__ORIGIN__', json.dumps(f'http://127.0.0.1:{server.server_port}'))
        try:
            result = subprocess.run(fixture_runner().aside_command(fixture_runner().find_aside()) + ['repl', script], capture_output=True, text=True, timeout=90)
            output = result.stdout + result.stderr
            lines = [line[len('TL_TESTS='):] for line in output.splitlines() if line.startswith('TL_TESTS=')]
            self.assertEqual(result.returncode, 0, output)
            self.assertTrue(lines, output)
            tests = json.loads(lines[0])
            self.assertEqual(len(tests), 5, output)
            self.assertTrue(all(test['ok'] for test in tests), output)
            print('Aside localhost regressions: location, expansion, duplicate fields, access block passed')
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__':
    unittest.main()
