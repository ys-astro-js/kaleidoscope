from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

GITHUB_SOURCE_ZIP_URL = "https://github.com/sony/clews/archive/refs/heads/main.zip"
ZENODO_CLEWS_ZIP_URL = "https://zenodo.org/records/15045900/files/clews.zip?download=1"

CLEWS_CHECKPOINT_NAME = "checkpoint_best.ckpt"
CLEWS_CONFIG_NAME = "configuration.yaml"
CLEWS_MANIFEST_NAME = "source_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official CLEWS assets for Kaleidoscope.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data") / "models" / "clews",
        help="Target CLEWS model directory, relative to backend/ by default.",
    )
    parser.add_argument(
        "--source-zip-url",
        default=GITHUB_SOURCE_ZIP_URL,
        help="Official sony/clews source zip URL.",
    )
    parser.add_argument(
        "--checkpoint-zip-url",
        default=ZENODO_CLEWS_ZIP_URL,
        help="Zenodo CLEWS checkpoint zip URL.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="clews-assets.") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source_zip = _download(args.source_zip_url, temp_dir / "clews-source.zip")
        checkpoint_zip = _download(args.checkpoint_zip_url, temp_dir / "clews-checkpoints.zip")

        source_dir = _extract_source(source_zip, model_dir)
        checkpoint_path, config_path = _extract_dvi_clews_checkpoint(checkpoint_zip, model_dir)
        _write_manifest(
            model_dir,
            {
                "source_url": args.source_zip_url,
                "checkpoint_url": args.checkpoint_zip_url,
                "source_dir": str(source_dir),
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_path),
            },
        )

    print(f"CLEWS assets ready under {model_dir.resolve()}")


def _download(url: str, target: Path) -> Path:
    print(f"Downloading {url}")
    urlretrieve(url, target)
    return target


def _extract_source(source_zip: Path, model_dir: Path) -> Path:
    source_root = model_dir / "source"
    if source_root.exists():
        shutil.rmtree(source_root)
    source_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source_zip) as archive:
        archive.extractall(source_root)

    source_candidates = [
        path
        for path in source_root.iterdir()
        if path.is_dir() and (path / "inference.py").exists() and (path / "models").exists()
    ]
    if not source_candidates:
        raise RuntimeError("Downloaded CLEWS source zip did not contain inference.py and models/.")
    return source_candidates[0]


def _extract_dvi_clews_checkpoint(checkpoint_zip: Path, model_dir: Path) -> tuple[Path, Path]:
    with zipfile.ZipFile(checkpoint_zip) as archive:
        checkpoint_member, config_member = _select_checkpoint_members(archive)
        checkpoint_path = model_dir / CLEWS_CHECKPOINT_NAME
        config_path = model_dir / CLEWS_CONFIG_NAME
        _extract_member_to_file(archive, checkpoint_member, checkpoint_path)
        _extract_member_to_file(archive, config_member, config_path)
    return checkpoint_path, config_path


def _select_checkpoint_members(archive: zipfile.ZipFile) -> tuple[str, str]:
    names = archive.namelist()
    checkpoint_names = [name for name in names if name.endswith(CLEWS_CHECKPOINT_NAME)]
    config_names = [name for name in names if name.endswith(CLEWS_CONFIG_NAME)]
    if not checkpoint_names or not config_names:
        raise RuntimeError("CLEWS checkpoint zip did not contain checkpoint_best.ckpt and configuration.yaml.")

    grouped = []
    for checkpoint_name in checkpoint_names:
        parent = str(Path(checkpoint_name).parent).replace("\\", "/")
        config_name = next(
            (name for name in config_names if str(Path(name).parent).replace("\\", "/") == parent),
            None,
        )
        if config_name is None:
            continue
        grouped.append((checkpoint_name, config_name))

    if not grouped:
        raise RuntimeError("No CLEWS checkpoint directory had both checkpoint_best.ckpt and configuration.yaml.")

    return sorted(grouped, key=lambda pair: _checkpoint_sort_key(pair[0]))[0]


def _checkpoint_sort_key(name: str) -> tuple[int, int, str]:
    lower = name.lower()
    is_dvi = "dvi" in lower
    is_clews = "clews" in lower
    return (0 if is_dvi and is_clews else 1, 0 if is_dvi else 1, lower)


def _extract_member_to_file(archive: zipfile.ZipFile, member: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination)


def _write_manifest(model_dir: Path, payload: dict[str, str]) -> None:
    with (model_dir / CLEWS_MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
