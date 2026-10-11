"""Exact-pinned, graph-only temporary materialization for the admitted FPN lane.

The stream parser stays shared with metadata-only discovery. Nothing is yielded
until the complete archive, every ordered member/PAX record, root, graph and
pipeline metadata match hosted discovery. The graph is staged without execution,
then rehashed from disk; all graph material is removed on failure or context exit.
This module installs no native dependency and performs no model construction.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import monotonic
from typing import Callable, Iterator

from . import weapons_oid_fpn_discovery as discovery

_IDENTITY_PATH = Path(__file__).with_name("data") / "weapons_oid_fpn_identity.json"
_IDENTITY_SHA256 = "f24ce64d8f685cdfb0f9d447db56570474868e34b373f3c64b02e9870f417678"
_MAX_IDENTITY_BYTES = 256 * 1024


def identity() -> dict:
    """Return a fresh defensive copy of the complete run-344 discovery identity."""
    if _IDENTITY_PATH.is_symlink():
        raise RuntimeError("pinned identity cannot be a symlink")
    with _IDENTITY_PATH.open("rb") as source:
        payload = source.read(_MAX_IDENTITY_BYTES + 1)
    if len(payload) > _MAX_IDENTITY_BYTES or hashlib.sha256(payload).hexdigest() != _IDENTITY_SHA256:
        raise RuntimeError("bundled FPN identity bytes differ from the reviewed pin")
    return json.loads(payload)


@contextmanager
def _work_parent(work_dir: str | Path) -> Iterator[Path]:
    configured = os.environ.get("RUNNER_TEMP")
    if not configured:
        raise RuntimeError("existing RUNNER_TEMP is required")
    runner, parent = Path(configured), Path(work_dir)
    if (not runner.is_absolute() or not runner.is_dir() or not parent.is_absolute()
            or ".." in parent.parts or ".." in runner.parts):
        raise RuntimeError("work directory must be absolute beneath existing RUNNER_TEMP")
    if any(item.is_symlink() for item in (runner, *runner.parents)):
        raise RuntimeError("RUNNER_TEMP cannot traverse symlinks")
    try:
        relative = parent.relative_to(runner)
    except ValueError as exc:
        raise RuntimeError("work directory must be beneath RUNNER_TEMP") from exc
    created = []
    try:
        current = runner
        for part in relative.parts:
            current = current / part
            if current.is_symlink() or current.exists() and not current.is_dir():
                raise RuntimeError("work directory cannot traverse symlinks or non-directories")
            if not current.exists():
                current.mkdir(mode=0o700)
                created.append(current)
        yield parent
    finally:
        for directory in reversed(created):
            try:
                directory.rmdir()
            except OSError:
                pass  # Preserve unrelated caller files; graph cleanup owns only its unique child.


@contextmanager
def verified_graph(work_dir: str | Path, *, opener: Callable | None = None,
                   clock: Callable = monotonic) -> Iterator[Path]:
    """Yield one verified graph path after full identity closure, for this context only."""
    deadline = clock() + discovery._TIME_LIMIT
    expected = identity()
    expected_graph = expected["graph"]
    expected_header = {key: expected_graph[key] for key in ("name", "original_name", "type", "size")}
    with _work_parent(work_dir) as parent:
        with tempfile.TemporaryDirectory(prefix="weapons-fpn-verified-", dir=parent) as temporary:
            graph_path = Path(temporary) / "frozen_inference_graph.pb"
            written = 0
            with graph_path.open("xb") as destination:
                def write_graph(header: dict, data: bytes) -> None:
                    nonlocal written
                    if (header != expected_header or type(data) is not bytes
                            or written + len(data) > expected_graph["size"]):
                        raise RuntimeError("streamed graph header/size differs from the exact pin")
                    if destination.write(data) != len(data):
                        raise RuntimeError("temporary graph write was incomplete")
                    written += len(data)

                actual = discovery._run(opener=opener, clock=clock, graph_sink=write_graph)
            if actual != expected or written != expected_graph["size"]:
                raise RuntimeError("complete archive/member/PAX/pipeline identity differs from the exact pin")
            digest, checked = hashlib.sha256(), 0
            with graph_path.open("rb") as materialized:
                while True:
                    discovery._remaining(deadline, clock)
                    data = materialized.read(discovery._CHUNK)
                    discovery._remaining(deadline, clock)
                    if not data:
                        break
                    checked += len(data)
                    if checked > expected_graph["size"]:
                        raise RuntimeError("temporary graph exceeds its pinned size")
                    digest.update(data)
            if checked != expected_graph["size"] or digest.hexdigest() != expected_graph["sha256"]:
                raise RuntimeError("materialized graph identity differs from the exact pin")
            discovery._remaining(deadline, clock)
            yield graph_path
