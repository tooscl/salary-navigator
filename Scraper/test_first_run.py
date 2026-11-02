"""First-run file preparation and CSV invariants; no installers or network."""
import base64
import io
import json
import plistlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).parent / "skills/collect-avito/scripts"
sys.path.insert(0, str(SCRIPTS))
import setup_workspace as setup
import preflight
import run as runner


class SetupTests(unittest.TestCase):
    def test_config_discovery_reuses_shared_file_without_overriding_explicit_or_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); local = root/'checkout/local.json'; shared = root/'private/collector.local.json'
            shared.parent.mkdir(); shared.write_text('{}')
            with patch.dict(preflight.os.environ, {}, clear=True), \
                    patch.object(preflight, 'DEFAULT_CONFIG', local), \
                    patch.object(preflight, 'saved_config_path', return_value=shared):
                self.assertEqual(preflight.default_config_path(), shared)
                local.parent.mkdir(); local.write_text('{}')
                self.assertEqual(preflight.default_config_path(), local)
                with patch.dict(preflight.os.environ, {'TL_AVITO_CONFIG': str(root/'explicit.json')}):
                    self.assertEqual(preflight.default_config_path(), root/'explicit.json')
                local.unlink(); shared.unlink()
                self.assertEqual(preflight.default_config_path(), local)

    def test_existing_routing_profile_registers_shared_config_without_server_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); private = root/'private'; old = root/'old-checkout.json'
            setup.prepare('happ', old, private)
            old.unlink()  # A fresh checkout has no settings pointer, but routing files survived.
            profile = (private/'avito-routing.json').read_bytes()
            link = (private/'avito-routing-link.txt').read_bytes()
            shared = private/'collector.local.json'
            result = setup.prepare('happ', shared, private)
            self.assertEqual(result['created'], [str(shared.resolve())])
            settings = json.loads(shared.read_text())
            self.assertEqual(settings['tunnel_mode'], 'happ')
            self.assertIsNone(settings['xray_config_path'])
            self.assertEqual(Path(settings['happ_routing_path']), (private/'avito-routing.json').resolve())
            self.assertEqual((private/'avito-routing.json').read_bytes(), profile)
            self.assertEqual((private/'avito-routing-link.txt').read_bytes(), link)
            self.assertFalse(result['live_route_verified'])
            self.assertFalse((private/'avito-tunnel.json').exists())

    def test_supplied_happ_tunnel_preserves_settings_and_registers_private_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'supplied.json'; directory = root/'private'
            original = {'dns': {'servers': ['localhost']}, 'custom': {'keep': True},
                'outbounds': [{'tag': 'proxy', 'protocol': 'vless',
                               'settings': {'example_secret': 'fixture-only'}},
                              {'tag': 'tl-avito-direct', 'protocol': 'blackhole'}],
                'routing': {'domainStrategy': 'AsIs', 'rules': [
                    {'type': 'field', 'network': 'tcp,udp', 'outboundTag': 'proxy'}]}}
            source.write_text(json.dumps(original)); before = source.read_bytes()
            result = setup.prepare_happ_tunnel(source, root/'local.json', directory)
            updated = json.loads((directory/'avito-tunnel.json').read_text())
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(json.loads((directory/'current-tunnel.original.json').read_text()), original)
            self.assertEqual(updated['outbounds'][:-1], original['outbounds'])
            self.assertEqual(updated['routing']['rules'][1:], original['routing']['rules'])
            self.assertEqual(updated['dns'], original['dns'])
            self.assertEqual(updated['custom'], original['custom'])
            self.assertEqual(preflight.check_xray(updated), [])
            self.assertEqual(json.loads((root/'local.json').read_text())['xray_config_path'], result['import_file'])
            self.assertFalse(result['live_route_verified'])
            self.assertEqual(result['import_kind'], 'full_connection')
            self.assertTrue(result['creates_connection'])
            self.assertNotIn('fixture-only', json.dumps(result))
            self.assertEqual(setup.prepare_happ_tunnel(source, root/'local.json', directory)['created'], [])
            if setup.os.name != 'nt':
                self.assertEqual((directory/'avito-tunnel.json').stat().st_mode & 0o777, 0o600)

    @unittest.skipIf(setup.os.name == 'nt', 'POSIX permissions')
    def test_supplied_happ_tunnel_permissions_at_creation_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'supplied.json'; directory = root/'private'
            source.write_text(json.dumps({'outbounds': [{'tag': 'direct', 'protocol': 'freedom'}]}))
            previous = setup.os.umask(0)
            try:
                setup.prepare_happ_tunnel(source, root/'local.json', directory)
            finally:
                setup.os.umask(previous)
            self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
            for name in ['avito-tunnel.json', 'current-tunnel.original.json']:
                self.assertEqual((directory/name).stat().st_mode & 0o777, 0o600)
                (directory/name).chmod(0o644)
            (root/'local.json').chmod(0o644)
            setup.prepare_happ_tunnel(source, root/'local.json', directory)
            for path in [directory/'avito-tunnel.json', directory/'current-tunnel.original.json', root/'local.json']:
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            directory.chmod(0o755)
            with self.assertRaisesRegex(setup.SetupError, '0700'):
                setup.prepare_happ_tunnel(source, root/'local.json', directory)

    @unittest.skipIf(setup.os.name == 'nt', 'POSIX symlinks')
    def test_supplied_happ_tunnel_rejects_symlink_destinations_before_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'supplied.json'; target = root/'target.json'
            source.write_text(json.dumps({'outbounds': [{'tag': 'direct', 'protocol': 'freedom'}]}))
            target.write_text('{}'); target.chmod(0o644)
            link = root/'linked-config.json'; link.symlink_to(target)
            with self.assertRaises(setup.SetupError):
                setup.prepare_happ_tunnel(source, link, root/'private')
            self.assertEqual(target.stat().st_mode & 0o777, 0o644)
            self.assertFalse((root/'private').exists())
            directory = root/'existing'; directory.mkdir()
            linked_dir = root/'linked-dir'; linked_dir.symlink_to(directory, target_is_directory=True)
            with self.assertRaises(setup.SetupError):
                setup.prepare_happ_tunnel(source, root/'local.json', linked_dir)
            self.assertEqual(list(directory.iterdir()), [])

    def test_supplied_happ_tunnel_reuses_plain_direct_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'supplied.json'; directory = root/'private'
            source.write_text(json.dumps({'outbounds': [{'tag': 'direct', 'protocol': 'freedom'}]}))
            setup.prepare_happ_tunnel(source, root/'local.json', directory)
            updated = json.loads((directory/'avito-tunnel.json').read_text())
            self.assertEqual(len(updated['outbounds']), 1)
            self.assertEqual(updated['routing']['rules'][0]['outboundTag'], 'direct')
            (directory/'avito-tunnel.json').write_text('user modified')
            before = (root/'local.json').read_bytes()
            with self.assertRaises(setup.SetupError):
                setup.prepare_happ_tunnel(source, root/'local.json', directory)
            self.assertEqual((directory/'avito-tunnel.json').read_text(), 'user modified')
            self.assertEqual((root/'local.json').read_bytes(), before)

    def test_supplied_happ_tunnel_rejects_unrecognized_structure_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'supplied.json'
            for data in [{'Name': 'routing profile'}, {'outbounds': []},
                         {'outbounds': [{'tag': 'p', 'protocol': 'vless'}], 'routing': {'rules': [None]}}]:
                source.write_text(json.dumps(data))
                with self.assertRaises(setup.SetupError):
                    setup.prepare_happ_tunnel(source, root/'local.json', root/'private')
                self.assertFalse((root/'private').exists())
                self.assertFalse((root/'local.json').exists())

    def test_local_happ_snapshot_prepares_private_copy_without_changing_preferences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prefs = root/'happ.plist'
            block = {'network': 'udp', 'port': '443', 'outboundTag': 'block'}
            original = {'inbounds': [{'protocol': 'socks', 'port': 10808}],
                'dns': {'servers': ['localhost']},
                'outbounds': [{'tag': 'proxy', 'protocol': 'vless', 'settings': {'id': 'fixture-secret'}},
                              {'tag': 'direct', 'protocol': 'freedom', 'settings': {'domainStrategy': 'UseIPv4'}},
                              {'tag': 'block', 'protocol': 'blackhole'}],
                'routing': {'rules': [block]}}
            prefs.write_bytes(plistlib.dumps({'connectedConfigJson': json.dumps(original)}))
            before = prefs.read_bytes()
            result = setup.prepare_local_happ(root/'local.json', root/'private', preferences=prefs, inspect_only=True)
            self.assertEqual(result['status'], 'ROUTING_MISSING')
            self.assertFalse((root/'private').exists())
            result = setup.prepare_local_happ(root/'local.json', root/'private', preferences=prefs)
            modified = json.loads(Path(result['import_file']).read_text())
            self.assertEqual(prefs.read_bytes(), before)
            self.assertEqual(modified['routing']['rules'][0], block)
            modified['routing']['rules'].pop(1)
            self.assertEqual(modified, original)
            self.assertFalse(result['live_route_verified'])
            self.assertFalse(result['freshness_confirmed'])
            self.assertNotIn('fixture-secret', json.dumps(result))
            self.assertEqual(setup.prepare_local_happ(root/'local.json', root/'private', preferences=prefs)['created'], [])

    def test_local_happ_already_routed_is_preserved_and_missing_cache_has_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prefs = root/'happ.plist'
            original = {'inbounds': [{'protocol': 'socks'}],
                'outbounds': [{'tag': 'd', 'protocol': 'freedom'}],
                'routing': {'rules': [{'type': 'field', 'domain': ['domain:avito.ru', 'domain:avito.st'], 'outboundTag': 'd'}]}}
            prefs.write_bytes(plistlib.dumps({'connectedConfigJson': original}))
            result = setup.prepare_local_happ(root/'local.json', root/'private', preferences=prefs)
            self.assertTrue(result['routing_present'])
            self.assertEqual(json.loads(Path(result['import_file']).read_text()), original)
            prefs.write_bytes(plistlib.dumps({'connectedConfigJson': 'invalid-secret'}))
            with self.assertRaises(setup.SetupError) as error:
                setup.prepare_local_happ(root/'new.json', root/'new-private', preferences=prefs)
            self.assertNotIn('invalid-secret', str(error.exception))
            self.assertFalse((root/'new-private').exists())
            self.assertFalse((root/'new.json').exists())

    def test_local_snapshot_reports_tcp_scope_without_claiming_udp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prefs = root/'happ.plist'
            original = {'inbounds': [{'protocol': 'socks'}],
                'outbounds': [{'tag': 'd', 'protocol': 'freedom'}, {'tag': 'p', 'protocol': 'vless'}],
                'routing': {'rules': [
                    {'type': 'field', 'network': 'tcp', 'domain': ['domain:avito.ru', 'domain:avito.st'], 'outboundTag': 'd'},
                    {'network': 'udp', 'outboundTag': 'p'}]}}
            prefs.write_bytes(plistlib.dumps({'connectedConfigJson': original}))
            result = setup.prepare_local_happ(root/'local.json', root/'private', preferences=prefs, inspect_only=True)
            self.assertEqual(result['status'], 'ROUTING_PRESENT')
            self.assertEqual(result['routing_scope'], 'tcp')
            self.assertFalse(result['live_route_verified'])
            self.assertFalse((root/'private').exists())

    def test_local_happ_discovery_requires_unambiguous_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(setup.sys, 'platform', 'darwin'), patch.object(setup.Path, 'home', return_value=root):
                with self.assertRaises(setup.SetupError):
                    setup.local_happ_snapshot()
                for app in ['group.su.ffg.happ.ruplus', 'group.su.ffg.happ.plus']:
                    p = root/'Library/Group Containers'/app/'Library/Preferences'/(app+'.plist')
                    p.parent.mkdir(parents=True); p.write_bytes(plistlib.dumps({'connectedConfigJson': {
                        'inbounds': [{'protocol': 'socks'}], 'outbounds': [{'tag': 'p', 'protocol': 'vless'}]}}))
                    if app.endswith('ruplus'):
                        self.assertEqual(setup.local_happ_snapshot()[1]['preferences_path'], str(p))
                with self.assertRaisesRegex(setup.SetupError, '--happ-preferences'):
                    setup.local_happ_snapshot()

    def test_runner_and_preflight_use_same_aside_outside_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);binary = root / '.local/bin/aside'
            binary.parent.mkdir(parents=True);binary.write_text('#!/bin/sh\n');binary.chmod(0o700)
            with patch.object(preflight.Path, 'home', return_value=root), \
                    patch.object(preflight.shutil, 'which', return_value=None), \
                    patch.object(runner, 'aside_command', side_effect=lambda executable:
                        [executable, '--host', 'local', '--account', 'u0', '--profile', 'connected']) as target, \
                    patch.object(runner.subprocess, 'Popen', side_effect=RuntimeError('test stop')) as command:
                self.assertEqual(preflight.find_aside(), str(binary))
                with self.assertRaisesRegex(RuntimeError, 'test stop'):
                    runner.browser({}, root, root)
                target.assert_called_once_with(str(binary))
                self.assertEqual(command.call_args.args[0], [str(binary), '--host', 'local',
                    '--account', 'u0', '--profile', 'connected', 'repl'])

    def test_runner_passes_explicit_config_to_its_preflight(self):
        with patch.object(runner.subprocess, 'run') as command:
            runner.preflight(Path('/private/custom/local.json'))
        self.assertEqual(command.call_args.args[0][-2:], ['--config', str(Path('/private/custom/local.json'))])
        with patch.object(runner.subprocess, 'run') as command:
            runner.preflight()
        self.assertNotIn('--config', command.call_args.args[0])

    def test_pipe_reader_preserves_bytes_and_reports_eof_without_select(self):
        stream = io.BufferedReader(io.BytesIO('TL_SESSION=пример\nTL_FINISHED\n'.encode()))
        chunks, thread = runner.pipe_chunks(stream)
        received = bytearray()
        while True:
            chunk = chunks.get(timeout=1)
            if chunk is None:
                break
            received.extend(chunk)
        thread.join(timeout=1)
        self.assertEqual(received.decode(), 'TL_SESSION=пример\nTL_FINISHED\n')

    def test_happ_files_are_importable_repeatable_and_static_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            settings = directory / "local.json"
            profile_dir = directory / "private"
            first = setup.prepare("happ", settings, profile_dir)
            link = (profile_dir / "avito-routing-link.txt").read_text().strip()
            decoded = json.loads(base64.b64decode(link.removeprefix("happ://routing/add/")))
            self.assertEqual(decoded, json.loads((profile_dir / "avito-routing.json").read_text()))
            self.assertEqual(preflight.check_happ(decoded), [])
            self.assertTrue(first["happ_import_required"])
            self.assertEqual(first["import_kind"], "routing_rules")
            self.assertFalse(first["creates_connection"])
            self.assertFalse(first["live_route_verified"])
            before = settings.stat().st_mtime_ns
            self.assertEqual(setup.prepare("happ", settings, profile_dir)["created"], [])
            self.assertEqual(settings.stat().st_mtime_ns, before)
            with patch.object(preflight, "find_browse_skill", return_value=SCRIPTS), \
                    patch.object(preflight.shutil, "which", return_value="aside"), \
                    patch.object(preflight, "aside_ready", return_value=True):
                checked = preflight.run_checks(settings, probe_aside=True, strict_routing=True)
            self.assertEqual(checked["status"], "LOCAL_CHECKS_OK")
            self.assertTrue(checked["can_collect"])
            self.assertFalse(checked["live_route_verified"])

    def test_existing_different_settings_are_not_reset_or_partly_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local.json"
            setup.prepare("disabled", path)
            previous = path.read_bytes()
            with self.assertRaises(ValueError):
                setup.prepare("happ", path, Path(tmp) / "private")
            self.assertEqual(path.read_bytes(), previous)
            self.assertFalse((Path(tmp) / "private").exists())

    def test_changed_happ_profile_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); private = directory / "private"
            setup.prepare("happ", directory / "settings.json", private)
            path = private / "avito-routing.json"
            path.write_text('{"Name":"user edited"}')
            with self.assertRaises(ValueError):
                setup.prepare("happ", directory / "settings.json", private)
            self.assertEqual(json.loads(path.read_text()), {"Name": "user edited"})

    def test_no_happ_artifacts_in_checkout(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, "ROOT", Path(tmp)):
            with self.assertRaises(ValueError):
                setup.prepare("happ", Path(tmp) / "local.json", Path(tmp) / "private")

    def test_happ_shadowing_and_missing_domain_fail(self):
        profile = setup.happ_profile()
        profile["RouteOrder"] = "block-proxy-direct"
        self.assertTrue(preflight.check_happ(profile))
        profile = setup.happ_profile();profile["DirectSites"].pop()
        self.assertTrue(preflight.check_happ(profile))

    def test_explicit_browse_path_and_invalid_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill = Path(tmp) / "SKILL.md";skill.write_text("name: browse")
            self.assertEqual(preflight.find_browse_skill(explicit=str(skill)), skill)
            self.assertIsNone(preflight.find_browse_skill(explicit=str(skill) + "missing"))


if __name__ == "__main__":
    unittest.main()
