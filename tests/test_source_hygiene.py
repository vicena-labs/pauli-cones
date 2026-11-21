"""The public release gate must reject credentials and internal artifacts."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_gate():
    path = Path(__file__).resolve().parents[1] / 'tools/check_source_hygiene.py'
    spec = importlib.util.spec_from_file_location('source_gate', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safe_source_fixture_passes(tmp_path):
    gate = load_gate()
    (tmp_path / 'demo.py').write_text('def answer(): return 42\n')
    result = gate.scan_tree(tmp_path)
    assert result['ok'], result['findings']


def test_rejects_synthetic_credential_without_disclosing_it(tmp_path):
    gate = load_gate()
    # Split the fixture's prefix to avoid publishing even a sample full token.
    fake = ''.join(('gh', 'p_', 'A' * 36))
    (tmp_path / 'example.py').write_text('KEY = ' + repr(fake))
    result = gate.scan_tree(tmp_path)
    assert not result['ok']
    assert any(item['reason'] == 'github_credential' for item in result['findings'])
    assert fake not in str(result)


def test_rejects_private_metadata_and_build_files(tmp_path):
    gate = load_gate()
    (tmp_path / 'SOURCE_PROVENANCE.json').write_text('{}')
    build = tmp_path / 'dist'
    build.mkdir()
    (build / 'unused.so').write_bytes(b'abc')
    report = gate.scan_tree(tmp_path)
    assert not report['ok']
    assert len(report['findings']) >= 2


def test_rejects_symlinks(tmp_path):
    gate = load_gate()
    original = tmp_path / 'original.txt'
    original.write_text('safe')
    (tmp_path / 'pointer.txt').symlink_to(original)
    report = gate.scan_tree(tmp_path)
    assert not report['ok']
    assert any(item['reason'] == 'symlink_not_reviewed' for item in report['findings'])


def git_stage(root, *names):
    import subprocess
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'add', '-f', '--', *names], check=True)


def test_scans_own_source_file(tmp_path):
    gate = load_gate()
    target = tmp_path / 'tools' / 'check_source_hygiene.py'
    target.parent.mkdir()
    fake = ''.join(('gh', 'p_', 'B' * 36))
    target.write_text('KEY = ' + repr(fake))
    result = gate.scan_tree(tmp_path)
    assert not result['ok']
    assert any(item['reason'] == 'github_credential' for item in result['findings'])
    assert fake not in str(result)


def test_tracked_checkout_excludes_git_metadata(tmp_path):
    gate = load_gate()
    (tmp_path / 'demo.py').write_text('answer = 42\n')
    git_stage(tmp_path, 'demo.py')
    result = gate.scan_tree(tmp_path, tracked_only=True)
    assert result['ok'], result['findings']
    assert result['text_files_checked'] == 1
    assert not gate.scan_tree(tmp_path)['ok']  # Git history must not enter a source archive.


def test_tracked_secret_is_rejected(tmp_path):
    gate = load_gate()
    fake = ''.join(('gh', 'p_', 'C' * 36))
    (tmp_path / 'example.py').write_text('KEY = ' + repr(fake))
    git_stage(tmp_path, 'example.py')
    result = gate.scan_tree(tmp_path, tracked_only=True)
    assert not result['ok']
    assert fake not in str(result)


def test_tracked_environment_file_is_rejected(tmp_path):
    gate = load_gate()
    (tmp_path / '.env').write_text('NOT_A_SECRET=1\n')
    git_stage(tmp_path, '.env')
    result = gate.scan_tree(tmp_path, tracked_only=True)
    assert not result['ok']
    assert any(f['reason'] == 'excluded_build_or_environment_path' for f in result['findings'])


def test_tracked_mode_requires_actual_repository_root(tmp_path):
    gate = load_gate()
    result = gate.scan_tree(tmp_path, tracked_only=True)
    assert not result['ok']
    assert result['findings'][0]['reason'] == 'source_selection_failed'


def test_missing_tracked_file_fails_closed(tmp_path):
    gate = load_gate()
    (tmp_path / 'demo.py').write_text('answer = 42\n')
    git_stage(tmp_path, 'demo.py')
    (tmp_path / 'demo.py').unlink()
    result = gate.scan_tree(tmp_path, tracked_only=True)
    assert not result['ok']
    assert any(f['reason'] == 'tracked_file_missing' for f in result['findings'])
