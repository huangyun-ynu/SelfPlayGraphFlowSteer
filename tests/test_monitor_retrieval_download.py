import importlib.util
from pathlib import Path


def test_progress_does_not_double_count_assembly_or_finished_files(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts/formal"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("monitor", scripts / "monitor_retrieval_download.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "FILES", [("repo", "rev", "part", 10, "hash")])
    (tmp_path / "part.partial").write_bytes(b"ab")
    ranges = tmp_path / "part.ranges"
    ranges.mkdir()
    (ranges / "2-5.chunk").write_bytes(b"cdef")
    (ranges / "2-5.tmp").write_bytes(b"cd")
    (tmp_path / "part.assembled").write_bytes(b"abcdef")
    assert module.progress(tmp_path) == {"part": 6}
    (tmp_path / "part.partial").write_bytes(b"abcdefghij")
    assert module.progress(tmp_path) == {"part": 10}
    (tmp_path / "part.partial").rename(tmp_path / "part")
    assert module.progress(tmp_path) == {"part": 10}
    assert not module.alive(999999999)
