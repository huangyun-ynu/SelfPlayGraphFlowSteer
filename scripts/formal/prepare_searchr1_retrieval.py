"""Download the fixed Search-R1 Wiki-18 corpus and its matching E5 index."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode, urlsplit

FILES = [
    ("wiki-18-e5-index", "a4d31160a035f30764604f4827cd8f1d0315eb86", "part_aa",
     42949672960, "a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023"),
    ("wiki-18-e5-index", "a4d31160a035f30764604f4827cd8f1d0315eb86", "part_ab",
     21609402413, "b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11"),
    ("wiki-18-corpus", "69c1c00ffe7c5554c68d8548355cb22e46aabc51", "wiki-18.jsonl.gz",
     5123307260, "7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db"),
]

MODELSCOPE_REVISIONS = {
    "wiki-18-e5-index": "66f7f27585673bd50d9dbf7d296a8c5959231fee",
    "wiki-18-corpus": "b861f07c146620f4456ddef013bb7196519b7512",
}


def download_command(repo, revision, name, partial, *, source, direct, resolve):
    if source == "modelscope":
        query = urlencode({"Revision": MODELSCOPE_REVISIONS[repo], "FilePath": name})
        url = f"https://modelscope.cn/api/v1/datasets/yamseyoung/{repo}/repo?{query}"
    else:
        url = f"https://huggingface.co/datasets/PeterJinGo/{repo}/resolve/{revision}/{name}"
    command = ["curl", "--silent", "--show-error", "--fail", "--location", "--connect-timeout", "15",
               "--speed-limit", "1024", "--speed-time", "120",
               "--continue-at", "-", "--output", str(partial)]
    if direct:
        command.extend(["--noproxy", "*"])
    for address in resolve:
        command.extend(["--resolve", address])
    return [*command, url]


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def extract_single_jsonl_tar(archive: Path, destination: Path) -> None:
    """Extract the sole JSONL member without trusting archive paths."""
    with tarfile.open(archive, mode="r:") as bundle:
        members = bundle.getmembers()
        files = [member for member in members if member.isfile()]
        unsafe = [member.name for member in members if not (member.isfile() or member.isdir())]
        if unsafe:
            raise ValueError(f"Corpus archive contains unsupported members: {unsafe}")
        if len(files) != 1 or not files[0].name.lower().endswith(".jsonl"):
            raise ValueError("Corpus archive must contain exactly one JSONL file")
        member = files[0]
        source = bundle.extractfile(member)
        if source is None:
            raise ValueError(f"Could not read corpus archive member: {member.name}")
        with source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, 8 * 1024 * 1024)
    if destination.stat().st_size != member.size:
        raise ValueError("Extracted corpus size does not match TAR metadata")


def invalidate_corpus_offsets(corpus: Path) -> None:
    offsets = corpus.with_suffix(corpus.suffix + ".offsets")
    offsets.unlink(missing_ok=True)
    offsets.with_suffix(".metadata.json").unlink(missing_ok=True)


def validate_range(headers, start, end, total, actual_size):
    values = [line.split(":", 1)[1].strip() for line in headers.splitlines()
              if line.lower().startswith("content-range:")]
    return values == [f"bytes {start}-{end}/{total}"] and actual_size == end - start + 1


def download_ranges(spec, partial, *, connections, resolve, storage_ips, chunk_size=16 * 1024 * 1024):
    repo, revision, name, size, digest = spec
    folder = partial.parent / (name + ".ranges")
    folder.mkdir(exist_ok=True)
    manifest_path = folder / "manifest.json"
    prefix = partial.stat().st_size if partial.exists() else 0
    manifest = {"size": size, "sha256": digest, "prefix": prefix, "chunk_size": chunk_size}
    if prefix > size:
        raise ValueError(f"Oversized existing prefix: {name}")
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"Range manifest differs from existing prefix: {name}")
    else:
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest))
        temporary.replace(manifest_path)

    location_lock = threading.Lock()
    location = ["", 0.0]

    def get_location():
        with location_lock:
            if location[0] and time.monotonic() - location[1] < 600:
                return location[0]
            url = download_command(repo, revision, name, partial, source="modelscope",
                                   direct=True, resolve=resolve)[-1]
            command = ["curl", "--noproxy", "*", "-fsS", "--connect-timeout", "15",
                       "--max-time", "30", "-o", "/dev/null", "-w", "%{redirect_url}"]
            for address in resolve:
                command += ["--resolve", address]
            result = subprocess.run([*command, url], capture_output=True, text=True)
            target = result.stdout.strip()
            host = urlsplit(target).hostname or ""
            if result.returncode or urlsplit(target).scheme != "https" or not host.endswith(".modelscope.cn"):
                raise RuntimeError("Could not obtain ModelScope storage URL")
            location[:] = [target, time.monotonic()]
            return target

    def fetch(bounds):
        number, start, end = bounds
        chunk = folder / f"{start:012d}-{end:012d}.chunk"
        if chunk.exists() and chunk.stat().st_size == end - start + 1:
            return chunk
        temporary = chunk.with_suffix(".tmp")
        headers = chunk.with_suffix(".headers")
        for attempt in range(20):
            try:
                target = get_location()
                host = urlsplit(target).hostname
                command = ["curl", "--noproxy", "*", "-fsS", "--connect-timeout", "15",
                           "--max-time", "900", "--speed-limit", "1024", "--speed-time", "60",
                           "--range", f"{start}-{end}", "--max-filesize", str(end-start+1),
                           "-D", str(headers), "-o", str(temporary), "-w", "%{http_code}"]
                if storage_ips:
                    ip = storage_ips[(number + attempt) % len(storage_ips)]
                    command += ["--resolve", f"{host}:443:{ip}"]
                # Signed URLs stay off the process command line and out of logs.
                result = subprocess.run([*command, "--config", "-"],
                                        input="url = " + json.dumps(target) + "\n",
                                        capture_output=True, text=True)
                valid = (result.returncode == 0 and result.stdout.strip() == "206"
                         and temporary.exists() and headers.exists()
                         and validate_range(headers.read_text(), start, end, size,
                                            temporary.stat().st_size))
                if valid:
                    temporary.replace(chunk)
                    print(f"Range ready {name} {start}-{end}", flush=True)
                    return chunk
                print(f"Retry {name} range {start}: HTTP {result.stdout.strip()} curl {result.returncode}", flush=True)
                with location_lock:
                    location[1] = 0.0
            except RuntimeError:
                print(f"Retry {name} range {start}: storage lookup failed", flush=True)
            finally:
                headers.unlink(missing_ok=True)
            time.sleep(min(30, 2 * (attempt + 1)))
        raise RuntimeError(f"Range download exhausted retries: {name} {start}-{end}")

    ranges = [(i, start, min(size - 1, start + chunk_size - 1))
              for i, start in enumerate(range(prefix, size, chunk_size))]
    print(f"Segmented {name}: preserving {prefix:,} bytes, {connections} connections", flush=True)
    with ThreadPoolExecutor(max_workers=connections) as pool:
        chunks = list(pool.map(fetch, ranges))
    assembled = partial.with_suffix(".assembled")
    with assembled.open("wb") as target:
        if prefix:
            with partial.open("rb") as source:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        for chunk in chunks:
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
    if assembled.stat().st_size != size or sha256(assembled) != digest:
        raise ValueError(f"Assembled download failed integrity check: {name}")
    assembled.replace(partial)
    shutil.rmtree(folder)


def prepare(output: Path, *, source="huggingface", direct=False, resolve=(), workers=1,
            connections=1, storage_ips=()):
    if not 1 <= workers <= 3:
        raise ValueError("workers must be between 1 and 3")
    download_source = source

    def download_one(spec):
        repo, revision, name, size, digest = spec
        path = output / name
        if not path.exists():
            partial = output / (name + ".partial")
            if connections > 1 and (not partial.exists() or partial.stat().st_size != size):
                if source != "modelscope" or not direct:
                    raise ValueError("Segmented downloads require ModelScope direct mode")
                download_ranges(spec, partial, connections=connections, resolve=resolve,
                                storage_ips=storage_ips)
            if not partial.exists() or partial.stat().st_size != size:
                print(f"Downloading {name}: {size:,} bytes (resumable)", flush=True)
                command = download_command(repo, revision, name, partial,
                                           source=download_source, direct=direct, resolve=resolve)
                # Restart curl to resume from the bytes retained by the previous attempt.
                for attempt in range(6):
                    result = subprocess.run(command, check=False)
                    if result.returncode == 0:
                        break
                    if attempt == 5:
                        raise subprocess.CalledProcessError(result.returncode, command)
                    print(f"Retrying {name} from its saved byte offset", flush=True)
                    time.sleep(3)
            if partial.stat().st_size != size or sha256(partial) != digest:
                raise ValueError(f"Download failed integrity check: {partial}")
            partial.replace(path)
        elif path.stat().st_size != size or sha256(path) != digest:
            raise ValueError(f"Existing file failed integrity check: {path}")
        print(f"Verified {name}", flush=True)
        return {"repo": f"PeterJinGo/{repo}", "revision": revision,
                "file": name, "bytes": size, "sha256": digest}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        sources = list(pool.map(download_one, FILES))

    index = output / "e5_Flat.index"
    index_size = FILES[0][3] + FILES[1][3]
    if index.exists():
        # Validate both ranges against the published split hashes.
        with index.open("rb") as source:
            for _, _, name, size, digest in FILES[:2]:
                remaining, actual = size, hashlib.sha256()
                while remaining:
                    chunk = source.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Truncated merged index")
                    actual.update(chunk)
                    remaining -= len(chunk)
                if actual.hexdigest() != digest:
                    raise ValueError(f"Merged index differs from {name}")
            if source.read(1):
                raise ValueError("Merged index has extra bytes")
    else:
        print("Merging verified index parts", flush=True)
        temporary = index.with_suffix(".partial")
        with temporary.open("wb") as target:
            for name in ("part_aa", "part_ab"):
                with (output / name).open("rb") as source:
                    shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        if temporary.stat().st_size != index_size:
            raise ValueError("Merged index size mismatch")
        temporary.replace(index)

    corpus = output / "wiki-18.jsonl"
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    corpus_is_tar = corpus.exists() and tarfile.is_tarfile(corpus)
    corpus_matches_manifest = (
        corpus.exists() and not corpus_is_tar
        and previous.get("corpus_sha256") == sha256(corpus)
    )
    if not corpus_matches_manifest:
        temporary = corpus.with_name(corpus.name + ".partial")
        temporary.unlink(missing_ok=True)
        if corpus_is_tar:
            # Older versions stopped after gzip decompression, leaving the inner
            # POSIX TAR at the path advertised as JSONL. Repair it in place.
            print("Extracting JSONL member from existing corpus TAR", flush=True)
            extract_single_jsonl_tar(corpus, temporary)
        else:
            print("Decompressing and extracting corpus (validates gzip CRC and TAR)", flush=True)
            archive = output / "wiki-18.tar.partial"
            archive.unlink(missing_ok=True)
            try:
                with gzip.open(output / "wiki-18.jsonl.gz", "rb") as source, archive.open("wb") as target:
                    shutil.copyfileobj(source, target, 8 * 1024 * 1024)
                extract_single_jsonl_tar(archive, temporary)
            finally:
                archive.unlink(missing_ok=True)
        temporary.replace(corpus)
        invalidate_corpus_offsets(corpus)
    manifest = {"sources": sources, "corpus_sha256": sha256(corpus),
                "corpus_bytes": corpus.stat().st_size, "index_bytes": index_size,
                "setting": "Search-R1 Wiki-18 shared fixed corpus for NQ and HotpotQA",
                "download_source": download_source, "direct": direct}
    temporary = manifest_path.with_suffix(".partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(f"Data ready: {manifest_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "state/formal-data/retrieval/searchr1")
    parser.add_argument("--source", choices=("huggingface", "modelscope"), default="huggingface")
    parser.add_argument("--direct", action="store_true", help="Bypass all proxy settings for downloads")
    parser.add_argument("--resolve", action="append", default=[], help="curl HOST:PORT:IP DNS override")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--connections", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--storage-ip", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args.output, source=args.source, direct=args.direct, resolve=args.resolve,
                workers=args.workers, connections=args.connections, storage_ips=args.storage_ip)


if __name__ == "__main__":
    main()
