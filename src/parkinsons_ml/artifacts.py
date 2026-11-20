"""Scoped local files and import paths for serialized model operations."""

import sys
from collections.abc import Generator
from contextlib import contextmanager
from tempfile import TemporaryDirectory


@contextmanager
def artifact_workspace() -> Generator[str]:
    """Clean downloads and SDK-added paths; callers must serialize model operations."""
    previous_path = list(sys.path)
    with TemporaryDirectory(prefix="parkinsons-artifacts-") as directory:
        try:
            yield directory
        finally:
            # Keep code/files alive through prediction, then remove their import
            # paths before deleting files. This does not sandbox deserialization.
            sys.path[:] = previous_path
