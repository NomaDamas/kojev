"""Contracts for gpu01 rsync and Slurm job templates."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SYNC = REPO / "scripts" / "sync.sh"
SMOKE = REPO / "scripts" / "slurm" / "smoke.sbatch"
TRAIN = REPO / "scripts" / "slurm" / "train.sbatch"
REMOTE_REPO = "/data2/jeffrey/kojev/repo"
HF_HOME = "/data2/jeffrey/kojev/hf-cache"
RUNS = "/data2/jeffrey/kojev/runs"
SHARED_CACHE = "/data2/jeffrey/kojev/uv-cache"
SHARED_VENV = "/data2/jeffrey/kojev/venv-shared"
FORBIDDEN_DESTS = ("gpu01:/data/", "gpu01:/data1/", "gpu01:/home")


def _text(path: Path) -> str:
    assert path.is_file(), f"missing {path.relative_to(REPO)}"
    return path.read_text(encoding="utf-8")


def _has_exclude(text: str, name: str) -> bool:
    needles = (
        f"--exclude {name}",
        f"--exclude={name}",
        f"--exclude '{name}'",
        f'--exclude "{name}"',
    )
    return any(needle in text for needle in needles)


def test_templates_share_one_package_cache_but_isolate_each_job_venv() -> None:
    # A per-job uv cache forces every job to re-download the multi-GB CUDA wheel
    # set, which exhausted the wall clock on jobs 13599/13601/13603. The cache
    # must be shared and warm; the venv must stay per-job to avoid the .venv race.
    for path in (SMOKE, TRAIN):
        text = _text(path)
        assert SHARED_CACHE in text
        assert f"{RUNS}/uv-cache-${{SLURM_JOB_ID}}" not in text
        assert f"export UV_PROJECT_ENVIRONMENT={RUNS}/venv-${{SLURM_JOB_ID}}" in text


def test_templates_hardlink_from_the_shared_cache_instead_of_copying() -> None:
    # Cache and per-job venv share the /data2 mount, so hardlinking avoids
    # re-copying torch (~10k files, ~2GB) into every job's venv. Dropping a job
    # venv only decrements link counts; shared cache entries survive.
    for path in (SMOKE, TRAIN):
        text = _text(path)
        assert "export UV_LINK_MODE=hardlink" in text
        assert "export UV_LINK_MODE=copy" not in text


def test_templates_prefer_a_prebuilt_shared_venv_and_never_mutate_it() -> None:
    # Installing torch (~10k files) into a per-job venv exceeded the smoke wall
    # clock on this NFS mount and exhausted job 13620. Jobs must reuse a
    # prebuilt shared environment and must never write to it.
    for path in (SMOKE, TRAIN):
        text = _text(path)
        assert SHARED_VENV in text
        assert "uv run --no-sync" in text
        assert "uv sync --no-dev" in text


def test_templates_skip_dev_dependencies_on_the_cluster() -> None:
    # basedpyright ships ~20k tiny typeshed stub files and drags in
    # nodejs-wheel-binaries. Unpacking those over the /data2 NFS mount dominated
    # the job wall clock and buys nothing at runtime: the cluster trains, it does
    # not lint.
    for path in (SMOKE, TRAIN):
        text = _text(path)
        assert "uv sync --no-dev" in text


def test_sync_script_exists_when_infra_is_present() -> None:
    # Given the repo checkout
    # When the sync helper is resolved
    # Then the file exists and is executable
    assert SYNC.is_file()
    assert SYNC.stat().st_mode & 0o111


def test_sync_rsyncs_repo_to_data2_when_excludes_are_applied() -> None:
    # Given scripts/sync.sh
    # When the rsync invocation is read
    # Then destination is gpu01:/data2/jeffrey/kojev/repo
    # and .git/.venv/data are excluded
    text = _text(SYNC)
    assert "rsync" in text
    assert f"gpu01:{REMOTE_REPO}" in text
    assert _has_exclude(text, ".git")
    assert _has_exclude(text, ".venv")
    assert _has_exclude(text, "data")
    for dest in FORBIDDEN_DESTS:
        assert dest not in text


def test_smoke_sbatch_requests_interactive_rtx6000_when_template_is_read() -> None:
    # Given scripts/slurm/smoke.sbatch
    # When SBATCH directives are read
    # Then partition/resources/GPU match the plan and GRES is not a wrong name
    text = _text(SMOKE)
    assert "#SBATCH --partition=interactive" in text
    assert "#SBATCH --gres=gpu:rtx6000:1" in text
    assert "#SBATCH --time=00:30:00" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=32G" in text
    assert "#SBATCH --gres=gpu:a100" not in text
    assert "#SBATCH --gres=gpu:1" not in text


def test_train_sbatch_requests_batch_rtx6000_when_template_is_read() -> None:
    text = _text(TRAIN)
    assert "#SBATCH --partition=batch" in text
    assert "#SBATCH --gres=gpu:rtx6000:1" in text
    assert "#SBATCH --time=24:00:00" in text
    assert "#SBATCH --cpus-per-task=12" in text
    assert "#SBATCH --mem=64G" in text
    assert "#SBATCH --gres=gpu:a100" not in text


def test_templates_export_hf_home_and_run_uv_when_body_is_read() -> None:
    # Given both Slurm templates
    # When the script bodies are read
    # Then paths, uv sync, and argument passthrough are present
    for path in (SMOKE, TRAIN):
        text = _text(path)
        cache_export = 'export UV_CACHE_DIR="${UV_CACHE_DIR:-' + SHARED_CACHE + '}"'
        assert f"export HF_HOME={HF_HOME}" in text
        assert f"export UV_PROJECT_ENVIRONMENT={RUNS}/venv-${{SLURM_JOB_ID}}" in text
        assert cache_export in text
        assert f"cd {REMOTE_REPO}" in text
        assert "uv sync --no-dev" in text
        assert 'uv run --no-sync "$@"' in text
        assert "/data/" not in text.replace("/data2/", "")
        assert "/data1/" not in text
