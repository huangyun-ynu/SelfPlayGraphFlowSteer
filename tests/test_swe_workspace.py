import os
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.swebench import CodeArtifactStore, SWEWorkspaceLifecycle, _git_safe_environment
from selfplay_graph_flowsteer.contracts import CodeArtifactRef


@pytest.mark.parametrize('foreign_owner', [False, True])
def test_real_git_workspace_patch_roundtrip_and_cleanup(tmp_path, foreign_owner):
    if foreign_owner and os.geteuid() != 0:
        pytest.skip('ownership regression needs root to create a foreign-owned fixture')
    cache = tmp_path / 'cache/example/repo'
    cache.mkdir(parents=True)
    def git(*args):
        with _git_safe_environment(cache) as overrides:
            return subprocess.check_output(['git', '-C', str(cache), *args],
                env={**os.environ, **overrides}, stderr=subprocess.STDOUT).decode().strip()
    git('init', '--quiet')
    (cache / 'sample.py').write_text('answer = 1\n')
    (cache / '.gitignore').write_text('__pycache__/\n')
    git('add', 'sample.py', '.gitignore')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
    commit = git('rev-parse', 'HEAD')
    if foreign_owner:
        for path in [cache, *cache.rglob('*')]:
            os.chown(path, 65534, 65534)
    life = SWEWorkspaceLifecycle(repo_cache_root=tmp_path/'cache', workspace_root=tmp_path/'workspaces',
        artifact_store=CodeArtifactStore(tmp_path/'artifacts'), test_profiles={'syntax': (sys.executable, '-m', 'py_compile')})
    life.bind_task(TaskSpec('fixture', 'Update the answer', metadata={'dataset':'swe_bench',
        'instance_id':'example__repo-1', 'repo':'example/repo', 'base_commit':commit}))
    life.begin_execution(agent_id='a', seed=0, revision=False)
    workspace = life._active.workspace
    if foreign_owner:
        for path in [workspace, *workspace.rglob('*')]:
            os.chown(path, 65534, 65534)
    changed = life.edit(dict(path='sample.py', operation='replace', workspace_version=0,
        expected_sha256=hashlib.sha256(b'answer = 1\n').hexdigest(),
        old_content='answer = 1', new_content='answer = 2'))
    tested = life.test('syntax', 'sample.py', workspace_version=changed['workspace_version'])
    assert tested['returncode'] == 0
    life.end_execution()
    result = life.result_for('a')
    assert result['workspace_completed'], result
    ref = CodeArtifactRef(**result['code_artifact_ref'])
    assert b'+answer = 2' in life.artifact_store.read(ref)
    assert not workspace.exists()
    life.set_visible_artifacts([ref])
    life.begin_execution(agent_id='b', seed=0, revision=True)
    applied = life.apply_artifact(ref.artifact_sha256, workspace_version=0)
    assert applied['changed_files'] == ['sample.py']
    assert (life._active.workspace/'sample.py').read_text() == 'answer = 2\n'
    life.end_execution()
    assert life.created_workspace_count == life.cleaned_workspace_count == 2
    assert not life.orphan_workspaces()
