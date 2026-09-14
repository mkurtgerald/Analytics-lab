"""Replay a bounded JSONL observation stream, not raw video. No network access."""

import argparse
import json
from pathlib import Path
import sys

from .temporal import Observation, PersonDownEngine


MAX_LINE_BYTES = 16384


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-id", required=True, help="Opaque non-secret source identifier")
    parser.add_argument("--session-id", required=True, help="Unique stream/tracker session identifier")
    args = parser.parse_args()
    line_number = 0
    try:
        engine = PersonDownEngine(args.source_id, args.session_id)
        with args.input.open("rb") as handle:
            while raw := handle.readline(MAX_LINE_BYTES + 1):
                line_number += 1
                if len(raw) > MAX_LINE_BYTES:
                    raise ValueError("oversized observation line")
                if not raw.strip():
                    continue
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("observation must be a JSON object")
                for event in engine.observe(Observation(**data)):
                    print(json.dumps(event, allow_nan=False, sort_keys=True))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError, RecursionError) as error:
        # Do not echo untrusted source data, URLs, paths, or credentials.
        print(f"Replay rejected at line {line_number} ({type(error).__name__}).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
