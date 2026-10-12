"""Download pinned PlantVillage metadata and extract only original color images."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def download(url: str, target: Path, expected_bytes: int | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected_bytes is None or target.stat().st_size == expected_bytes):
        print(f"Already downloaded: {target.name}", flush=True)
        return
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "PlantDiseaseLearningProject/1.0"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0
                total = expected_bytes or (offset + int(response.headers.get("Content-Length", "0")))
                mode = "ab" if offset else "wb"
                received = offset
                last_report = time.monotonic()
                with partial.open(mode) as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_report > 10:
                            percent = f" ({received / total:.1%})" if total else ""
                            print(f"{target.name}: {received / 1024**2:.0f} MiB{percent}", flush=True)
                            last_report = time.monotonic()
            if expected_bytes is not None and partial.stat().st_size != expected_bytes:
                raise OSError(f"Unexpected archive size: {partial.stat().st_size}")
            partial.replace(target)
            print(f"Downloaded: {target.name}", flush=True)
            return
        except (urllib.error.URLError, OSError) as error:
            if attempt == 3:
                raise
            print(f"Download interrupted: {error}; retrying from saved partial file", flush=True)
            time.sleep(2)


def main() -> None:
    config = json.loads((ROOT / "configs/dataset.json").read_text())
    data_dir = ROOT / "data"
    source_dir = data_dir / "source"
    archive = data_dir / "downloads/plantvillage.zip"
    base_url = config["base_url"]
    for filename in config["metadata_files"]:
        download(f"{base_url}/{filename}", source_dir / filename)
    needed = config["archive_bytes"] + 1024**3
    if not archive.exists() and shutil.disk_usage(ROOT).free < needed:
        raise OSError("Insufficient free disk space for the archive and color images")
    download(f"{base_url}/data.zip", archive, config["archive_bytes"])
    with archive.open("rb") as archive_file:
        digest = hashlib.file_digest(archive_file, "sha256").hexdigest()
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive) as zipped:
        images = []
        for entry in zipped.infolist():
            path = PurePosixPath(entry.filename)
            parts = path.parts
            if "raw" not in parts:
                continue
            start = parts.index("raw")
            relative = PurePosixPath(*parts[start + 1:])
            if relative.parts[:1] != ("color",) or entry.is_dir():
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive entry: {entry.filename}")
            if relative.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            images.append((entry, relative))
        if not images:
            raise ValueError("Archive contains no recognizable raw/color images")
        for index, (entry, relative) in enumerate(images, start=1):
            target = raw_dir.joinpath(*relative.parts)
            if not target.exists() or target.stat().st_size != entry.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                with zipped.open(entry) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(target)
            extracted += 1
            if index % 5000 == 0:
                print(f"Extracted {index}/{len(images)} color images", flush=True)
    provenance = {**config, "archive_sha256": digest, "extracted_color_images": extracted}
    (source_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Ready: {extracted} color images; archive SHA-256 {digest}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Download failed: {error}", file=sys.stderr, flush=True)
        raise
