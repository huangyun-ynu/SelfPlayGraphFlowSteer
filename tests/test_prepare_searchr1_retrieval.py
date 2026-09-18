import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest


def gzip_tar_jsonl(corpus: bytes, *, extra: bytes | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        member = tarfile.TarInfo("nested/wiki_dump.jsonl")
        member.size = len(corpus)
        bundle.addfile(member, io.BytesIO(corpus))
        if extra is not None:
            member = tarfile.TarInfo("extra.jsonl")
            member.size = len(extra)
            bundle.addfile(member, io.BytesIO(extra))
    return buffer.getvalue()


@pytest.mark.parametrize("workers", [1, 3])
def test_verified_parts_merge_in_order_and_corruption_is_rejected(tmp_path: Path, workers):
    path = Path(__file__).resolve().parents[1] / "scripts/formal/prepare_searchr1_retrieval.py"
    spec = importlib.util.spec_from_file_location("prepare_searchr1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corpus = b'{"contents":"synthetic"}\n'
    parts = {"part_aa": b"first", "part_ab": b"second",
             "wiki-18.jsonl.gz": gzip_tar_jsonl(corpus)}
    module.FILES = []
    for name, data in parts.items():
        (tmp_path / name).write_bytes(data)
        module.FILES.append(("fixture", "pinned", name, len(data), hashlib.sha256(data).hexdigest()))
    module.prepare(tmp_path, workers=workers)
    assert (tmp_path / "e5_Flat.index").read_bytes() == b"firstsecond"
    assert (tmp_path / "wiki-18.jsonl").read_bytes() == corpus
    assert json.loads((tmp_path / "manifest.json").read_text())["index_bytes"] == 11
    module.prepare(tmp_path, workers=workers)
    (tmp_path / "e5_Flat.index").write_bytes(b"wrongsecond")
    with pytest.raises(ValueError, match="differs from part_aa"):
        module.prepare(tmp_path, workers=workers)


def test_repairs_legacy_tar_mislabeled_as_jsonl(tmp_path: Path):
    path = Path(__file__).resolve().parents[1] / "scripts/formal/prepare_searchr1_retrieval.py"
    spec = importlib.util.spec_from_file_location("prepare_searchr1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corpus = b'{"contents":"first"}\n{"contents":"second"}\n'
    compressed = gzip_tar_jsonl(corpus)
    archive = gzip.decompress(compressed)
    parts = {"part_aa": b"first", "part_ab": b"second", "wiki-18.jsonl.gz": compressed}
    module.FILES = []
    for name, data in parts.items():
        (tmp_path / name).write_bytes(data)
        module.FILES.append(("fixture", "pinned", name, len(data), hashlib.sha256(data).hexdigest()))
    (tmp_path / "wiki-18.jsonl").write_bytes(archive)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "corpus_sha256": hashlib.sha256(archive).hexdigest()}))
    (tmp_path / "wiki-18.jsonl.offsets").write_bytes(b"stale")
    (tmp_path / "wiki-18.jsonl.metadata.json").write_text("{}")

    module.prepare(tmp_path)

    assert (tmp_path / "wiki-18.jsonl").read_bytes() == corpus
    assert not (tmp_path / "wiki-18.jsonl.offsets").exists()
    assert not (tmp_path / "wiki-18.jsonl.metadata.json").exists()


def test_rejects_corpus_archive_with_multiple_jsonl_files(tmp_path: Path):
    path = Path(__file__).resolve().parents[1] / "scripts/formal/prepare_searchr1_retrieval.py"
    spec = importlib.util.spec_from_file_location("prepare_searchr1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compressed = gzip_tar_jsonl(b'{"contents":"first"}\n', extra=b'{"contents":"extra"}\n')
    parts = {"part_aa": b"first", "part_ab": b"second", "wiki-18.jsonl.gz": compressed}
    module.FILES = []
    for name, data in parts.items():
        (tmp_path / name).write_bytes(data)
        module.FILES.append(("fixture", "pinned", name, len(data), hashlib.sha256(data).hexdigest()))

    with pytest.raises(ValueError, match="exactly one JSONL"):
        module.prepare(tmp_path)


def test_modelscope_direct_command_preserves_resume_and_pins_revision(tmp_path: Path):
    path = Path(__file__).resolve().parents[1] / "scripts/formal/prepare_searchr1_retrieval.py"
    spec = importlib.util.spec_from_file_location("prepare_searchr1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.download_command("wiki-18-e5-index", "original", "part_aa",
                                      tmp_path / "part_aa.partial", source="modelscope",
                                      direct=True, resolve=["modelscope.cn:443:39.99.133.195"])
    assert command[command.index("--noproxy") + 1] == "*"
    assert command[command.index("--continue-at") + 1] == "-"
    assert command[command.index("--resolve") + 1] == "modelscope.cn:443:39.99.133.195"
    url = urlsplit(command[-1])
    assert url.hostname == "modelscope.cn"
    assert parse_qs(url.query) == {
        "Revision": [module.MODELSCOPE_REVISIONS["wiki-18-e5-index"]], "FilePath": ["part_aa"]}


@pytest.mark.parametrize("bad_range", [False, True])
def test_segmented_download_preserves_prefix_and_validates_ranges(tmp_path, monkeypatch, bad_range):
    path = Path(__file__).resolve().parents[1] / "scripts/formal/prepare_searchr1_retrieval.py"
    spec = importlib.util.spec_from_file_location("prepare_searchr1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = b"abcdefghijklmnopqrstuvwxyz"
    partial = tmp_path / "part_aa.partial"
    partial.write_bytes(data[:5])
    folder = tmp_path / "part_aa.ranges"
    folder.mkdir()
    digest = hashlib.sha256(data).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(
        {"size": len(data), "sha256": digest, "prefix": 5, "chunk_size": 7}))
    (folder / "000000000005-000000000011.chunk").write_bytes(data[5:12])
    requested = []

    def run(command, **kwargs):
        assert command[command.index("--noproxy") + 1] == "*"
        if "--range" not in command:
            return SimpleNamespace(returncode=0, stdout="https://cdn-lfs-cn-1.modelscope.cn/test")
        start, end = map(int, command[command.index("--range") + 1].split("-"))
        requested.append(start)
        assert "url = " in kwargs["input"]
        Path(command[command.index("-o") + 1]).write_bytes(data[start:end+1])
        Path(command[command.index("-D") + 1]).write_text(
            f"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {0 if bad_range else start}-{end}/{len(data)}\r\n")
        return SimpleNamespace(returncode=0, stdout="206")

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    args = (("wiki-18-e5-index", "original", "part_aa", len(data), digest), partial)
    options = {"connections": 3, "resolve": (), "storage_ips": ["127.0.0.1"], "chunk_size": 7}
    if bad_range:
        with pytest.raises(RuntimeError, match="exhausted retries"):
            module.download_ranges(*args, **options)
        assert partial.read_bytes() == data[:5]
    else:
        module.download_ranges(*args, **options)
        assert partial.read_bytes() == data
        assert not folder.exists()
    assert 5 not in requested
