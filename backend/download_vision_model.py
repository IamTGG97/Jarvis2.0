"""Download the object detection ONNX weights and verify their SHA-256.

Usage (from the repository root):

    .\\.venv\\Scripts\\python.exe backend\\download_vision_model.py
    .\\.venv\\Scripts\\python.exe backend\\download_vision_model.py --model yolox_nano
    .\\.venv\\Scripts\\python.exe backend\\download_vision_model.py --model all

Files go to backend/models/ (git-ignored) or VISION_MODELS_DIR. A file is only
moved into place after its checksum matches, so a partial or tampered download
never ends up where the backend loads it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

import vision


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, force: bool = False) -> Path:
    spec = vision.model_spec(name)
    target = vision.model_path(name)
    expected = str(spec["sha256"])
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file() and not force:
        if sha256_of(target) == expected:
            print(f"{name}: already present and verified at {target}")
            return target
        print(f"{name}: existing file has the wrong checksum, downloading again")

    partial = target.with_suffix(target.suffix + ".part")
    print(f"{name}: downloading {spec['url']}")
    digest = hashlib.sha256()
    with httpx.stream("GET", str(spec["url"]), follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_bytes(1 << 20):
                handle.write(chunk)
                digest.update(chunk)

    actual = digest.hexdigest()
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: SHA-256 mismatch (expected {expected}, got {actual}); file discarded.")
    partial.replace(target)
    print(f"{name}: saved and verified ({target.stat().st_size:,} bytes) at {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default=vision.model_name(),
        choices=sorted(vision.MODELS) + ["all"],
        help="model to download (default: VISION_MODEL or yolox_tiny)",
    )
    parser.add_argument("--force", action="store_true", help="download even if a verified file exists")
    args = parser.parse_args(argv)
    names = sorted(vision.MODELS) if args.model == "all" else [args.model]
    try:
        for name in names:
            download(name, force=args.force)
    except (httpx.HTTPError, OSError, RuntimeError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
