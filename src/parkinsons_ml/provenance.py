"""Validate source hashes in Git-free images without pretending they are checkouts."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def source_hashes(root: Path) -> dict[str, str]:
    paths = [
        root / name
        for name in (
            "uv.lock",
            "pyproject.toml",
            "requirements.txt",
            "params.yaml",
            "dvc.yaml",
            "dvc.lock",
            "documents/data-bundle.json",
            "dataset.dvc",
            "Dockerfile",
            ".dockerignore",
            "compose.yaml",
        )
    ]
    paths.extend(sorted((root / "src/parkinsons_ml").rglob("*.py")))
    if not (root / "src/parkinsons_ml/tracking.py").is_file():
        raise ValueError("Image source layout is incomplete")
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def write_provenance(root: Path, revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Build requires a full Git commit SHA")
    manifest = {"format_version": 1, "build_revision": revision, "sha256": source_hashes(root)}
    (root / "build-provenance.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")


def image_lineage(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "build-provenance.json").read_bytes())
    if (
        not isinstance(manifest, dict)
        or manifest.get("format_version") != 1
        or not isinstance(manifest.get("build_revision"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", manifest["build_revision"])
        or manifest.get("sha256") != source_hashes(root)
    ):
        raise ValueError("Image provenance does not match the deployed files")
    # A supplied revision is build metadata, not proof that Git inspected the image.
    return {"source": "image-build", "build_revision": manifest["build_revision"], "sha256": manifest["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    write_provenance(Path(__file__).resolve().parents[2], args.revision)


if __name__ == "__main__":
    main()
