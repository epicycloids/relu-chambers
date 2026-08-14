"""Acquire external datasets used by the paper's experiment drivers.

The repository does not redistribute dataset payloads.  This module downloads
YearPredictionMSD from UCI, verifies the archive and source text, and converts
the comma-separated file to the ``data_cache/msd.npz`` layout expected by
``experiments.real_data``.

    python -m experiments.prepare_data msd
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MSD_URL = "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"
MSD_ZIP_SHA256 = "06f801af323bb7798e800583acce4ea1ed2697ac12c23f4424aea0a7a3d09e11"
MSD_TEXT_SHA256 = "4b6f8e50235b359e01689ae7fb33ad0f89677e9a15f25f3d6259327a6bb927bb"
MSD_SHAPE = (515_345, 91)
MSD_X_SHA256 = "d49d72f5a7149eb1c053021327c67dbd626af5ace8bc43fdf50670df39f0d560"
MSD_Y_SHA256 = "b8941502808b8c32a22be3474b469b43bffbcde9209a5376f65634dd9f5c4792"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_digest(path: Path, expected: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {observed}"
        )


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _validate_msd_cache(path: Path) -> None:
    with np.load(path, allow_pickle=False) as cached:
        if set(cached.files) != {"X", "y"}:
            raise ValueError(f"expected X and y arrays in {path}, got {cached.files}")
        X, y = cached["X"], cached["y"]
        if X.shape != (MSD_SHAPE[0], MSD_SHAPE[1] - 1) or X.dtype != np.float32:
            raise ValueError(
                f"unexpected cached MSD feature array: {X.shape} {X.dtype}"
            )
        if y.shape != (MSD_SHAPE[0],) or y.dtype != np.float64:
            raise ValueError(f"unexpected cached MSD target array: {y.shape} {y.dtype}")
        if _array_digest(X) != MSD_X_SHA256 or _array_digest(y) != MSD_Y_SHA256:
            raise ValueError(f"cached MSD array digest mismatch in {path}")


def _download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "relu-chambers-data-preparation/0.1"}
    )
    with urllib.request.urlopen(request) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    partial.replace(destination)


def _extract_msd_text(archive: Path, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    with zipfile.ZipFile(archive) as zf:
        member = zf.getinfo("YearPredictionMSD.txt")
        with zf.open(member) as src, partial.open("wb") as out:
            shutil.copyfileobj(src, out, length=1024 * 1024)
    partial.replace(destination)


def prepare_msd(cache_dir: Path, source_text: Path | None = None) -> Path:
    """Create ``msd.npz`` and return its path.

    ``source_text`` is an optional already-downloaded UCI text file.  It is
    primarily useful on machines that maintain a shared dataset cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / "msd.npz"
    if output.exists():
        _validate_msd_cache(output)
        print(f"verified existing {output}")
        return output

    if source_text is None:
        archive = cache_dir / "yearpredictionmsd.zip"
        text_path = cache_dir / "YearPredictionMSD.txt"
        if archive.exists():
            _require_digest(archive, MSD_ZIP_SHA256)
        else:
            _download(MSD_URL, archive)
            _require_digest(archive, MSD_ZIP_SHA256)
        if text_path.exists():
            _require_digest(text_path, MSD_TEXT_SHA256)
        else:
            _extract_msd_text(archive, text_path)
            _require_digest(text_path, MSD_TEXT_SHA256)
    else:
        text_path = source_text.resolve()
        _require_digest(text_path, MSD_TEXT_SHA256)

    print(f"parsing {text_path}")
    rows = np.loadtxt(text_path, delimiter=",", dtype=np.float32)
    if rows.shape != MSD_SHAPE:
        raise ValueError(f"expected MSD shape {MSD_SHAPE}, got {rows.shape}")
    y = rows[:, 0].astype(np.float64)
    X = np.ascontiguousarray(rows[:, 1:])

    partial = output.with_suffix(".npz.part")
    with partial.open("wb") as f:
        np.savez_compressed(f, X=X, y=y)
    partial.replace(output)
    print(f"wrote {output}: X={X.shape} {X.dtype}, y={y.shape} {y.dtype}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("msd",))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data_cache",
        help="dataset cache directory (default: repository data_cache/)",
    )
    parser.add_argument(
        "--source-text",
        type=Path,
        help="use an existing YearPredictionMSD.txt instead of downloading",
    )
    args = parser.parse_args()
    prepare_msd(args.cache_dir, source_text=args.source_text)


if __name__ == "__main__":
    main()
