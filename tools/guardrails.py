"""Small, offline development preflight and CI controls; not a security sandbox.

Policy decisions use a fresh, tool-verified snapshot supplied by the worker.
No network requests, purchases, model downloads, media access, or mutations.
The workflow uses JSON (a YAML subset) so validation needs no dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "mkurtgerald/Analytics-lab"
ACTIONS = {
    "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
    "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
}
CEILINGS = {
    "max_active_implementation_prs": 1,
    "max_unchanged_ci_retries": 1,
    "max_ci_dispatches_per_session": 2,
    "max_donor_candidates_per_component": 3,
    "replan_after_no_progress_sessions": 2,
    "escalate_after_no_progress_sessions": 3,
    "max_test_job_minutes": 5,
    "max_gate_job_minutes": 1,
}
DENIED = (
    "paid_spend_authorized", "production_release_authorized",
    "cross_repository_writes_authorized", "self_hosted_runners_authorized",
    "license_changes_authorized", "accuracy_claims_from_synthetic_tests_authorized",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def validate_policy(policy: dict) -> None:
    require(type(policy) is dict, "Policy must be an object")
    require(type(policy.get("schema_version")) is int and policy["schema_version"] == 1,
            "Unsupported policy version")
    require(policy.get("repository") == REPOSITORY, "Repository scope mismatch")
    for key, ceiling in CEILINGS.items():
        value = policy.get(key)
        require(type(value) is int and 1 <= value <= ceiling, "Invalid policy limit: " + key)
    require(policy["replan_after_no_progress_sessions"] < policy["escalate_after_no_progress_sessions"],
            "Replan must precede escalation")
    for key in DENIED:
        require(policy.get(key) is False, "Approval boundary changed: " + key)


def needs_runtime_tests(paths: list[str] | None) -> bool:
    """Skip runtime tests ONLY for a nonempty, verified Markdown-only diff.

    Control files (AGENTS, LICENSE, workflow, policy), unknown diffs and new
    file types always receive the complete tests. Deleted paths also count.
    """
    if not paths:
        return True
    for name in paths:
        if not isinstance(name, str) or not name or "\\" in name:
            return True
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            return True
        documentation = name in {"README.md", "CONTRIBUTING.md"} or (
            name.startswith("docs/") and path.suffix == ".md"
        )
        if not documentation:
            return True
    return False


def git_changes(base: str, head: str, pull_request: bool) -> list[str] | None:
    # Invalid/missing history costs a full test run; it must never skip tests.
    if any(not re.fullmatch(r"[0-9a-fA-F]{40}", ref) or set(ref) == {"0"}
           for ref in (base, head)):
        return None
    refs = [base + "..." + head] if pull_request else [base, head]
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--no-renames", "-z", *refs, "--"],
            cwd=ROOT, capture_output=True, check=True, timeout=20,
        )
        return [p.decode("utf-8") for p in result.stdout.split(b"\0") if p]
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def validate_workflow(workflow: dict, policy: dict) -> None:
    """Validate the supported CI envelope, not arbitrary code or data safety."""
    validate_policy(policy)
    require(workflow.get("on") == {"pull_request": {}, "push": {"branches": ["main"]}},
            "Unexpected CI trigger or filters")
    require(workflow.get("permissions") == {"contents": "read"}, "CI must be read-only")
    require(workflow.get("concurrency") == {
        "group": "analytics-ci-${{ github.event.pull_request.number || github.ref }}",
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }, "CI concurrency changed")
    jobs = workflow.get("jobs", {})
    require(set(jobs) == {"linux", "windows", "quality-gate"}, "Unexpected CI jobs")
    for name, job in jobs.items():
        runner = "windows-2022" if name == "windows" else "ubuntu-24.04"
        require(job.get("runs-on") == runner, "Unapproved runner")
        limit = policy["max_gate_job_minutes" if name == "quality-gate" else "max_test_job_minutes"]
        timeout = job.get("timeout-minutes")
        require(type(timeout) is int and 1 <= timeout <= limit, "CI timeout exceeds budget")
        require("strategy" not in job and "permissions" not in job, "CI fan-out or permission override")
        require("continue-on-error" not in job, "Fail-open CI job")
        for step in job.get("steps", []):
            require("continue-on-error" not in step, "Fail-open CI step")
            if "uses" in step:
                require(step["uses"] in ACTIONS, "Unapproved or unpinned CI action")
                if step["uses"].startswith("actions/checkout@"):
                    require(step.get("with", {}).get("persist-credentials") is False,
                            "Checkout credentials must not persist")
    for name in ("linux", "windows"):
        steps = jobs[name].get("steps", [])
        for command in (
            "python -m unittest discover -s tests -v",
            "python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id ci-fixture",
        ):
            found = [step for step in steps if step.get("run") == command]
            require(len(found) == 1, "Required regression or replay command missing")
            expected_if = "${{ steps.scope.outputs.run_full_tests == 'true' }}" if name == "linux" else None
            require(found[0].get("if") == expected_if, "Regression condition changed")
    linux_steps = jobs["linux"].get("steps", [])
    glyph_steps = [step for step in linux_steps if step.get("name") == "Bounded CC0 LPR glyph comparison evidence"]
    require(len(glyph_steps) == 1, "LPR glyph Linux evidence step missing")
    glyph = glyph_steps[0]
    require(glyph.get("if") == "${{ steps.scope.outputs.run_full_tests == 'true' && github.event_name == 'pull_request' && startsWith(github.head_ref, 'evidence/lpr-ocr-exact-glyph-') }}",
            "LPR glyph evidence must be restricted to the reviewed PR branch")
    glyph_run = glyph.get("run", "")
    for command in (
        "opencv-python-headless==4.12.0.88",
        "python -m unittest tests.test_lpr_ocr_glyph -v",
        "python -m analytics_lab.lpr_ocr_glyph_evidence",
    ):
        require(command in glyph_run, "LPR glyph evidence command missing")

    require(any(step.get("id") == "scope" and step.get("run") == "python tools/guardrails.py ci"
                and "if" not in step for step in linux_steps), "Scope preflight missing")
    require(any(step.get("run") == "python -m unittest discover -s tests -p test_guardrails.py -v"
                and "if" not in step for step in linux_steps), "Always-on guardrail tests missing")
    exact_steps = [step for step in jobs["windows"].get("steps", [])
                   if step.get("name") == "Bounded exact Tesseract 5.5.3 LPR OCR evidence"]
    require(len(exact_steps) == 1, "Exact Tesseract LPR evidence step missing")
    exact = exact_steps[0]
    require(exact.get("if") == "${{ github.event_name == 'pull_request' && startsWith(github.head_ref, 'evidence/lpr-ocr-exact-') && !startsWith(github.head_ref, 'evidence/lpr-ocr-exact-glyph-') }}",
            "Exact Tesseract lane must exclude the Linux glyph branch")
    require("lpr_ocr_glyph" not in exact.get("run", ""),
            "Glyph evidence must not execute in the Windows Tesseract lane")
    require(jobs["windows"].get("needs") == "linux", "Linux must precede Windows")
    require(jobs["windows"].get("if") == "${{ needs.linux.outputs.run_full_tests == 'true' }}",
            "Windows scope condition changed")
    gate = jobs["quality-gate"]
    require(gate.get("needs") == ["linux", "windows"] and gate.get("if") == "${{ always() }}",
            "Final gate must evaluate every result")


def preflight(snapshot: dict, policy: dict) -> dict:
    """Advise the worker from verified counts; this does not authenticate them.

    Counters are counts BEFORE the proposed action. A retry count of 1 means
    one unchanged retry already occurred, so another unchanged retry is denied.
    """
    validate_policy(policy)
    require(type(snapshot) is dict and snapshot.get("repository") == REPOSITORY,
            "Snapshot repository mismatch")
    require(snapshot.get("live_state_verified") is True, "Fresh live verification required")
    names = ("open_implementation_prs", "active_runs_for_head", "unchanged_ci_retries",
             "ci_dispatches_this_session", "sessions_without_progress")
    for name in names:
        require(type(snapshot.get(name)) is int and snapshot[name] >= 0,
                "Invalid snapshot count: " + name)
    action = snapshot.get("action")
    require(action in {"implement", "dispatch_ci", "retry_ci", "research"}, "Unknown action")
    reasons = []
    if snapshot["open_implementation_prs"] > policy["max_active_implementation_prs"]:
        reasons.append("Reconcile duplicate implementation PRs; do not create another")
    if snapshot["active_runs_for_head"] and action != "research":
        reasons.append("Current head is queued/running; preserve it and avoid duplicate work")
    if action in {"dispatch_ci", "retry_ci"}:
        if snapshot["ci_dispatches_this_session"] >= policy["max_ci_dispatches_per_session"]:
            reasons.append("CI dispatch budget exhausted; diagnose locally")
        if action == "retry_ci" and snapshot["unchanged_ci_retries"] >= policy["max_unchanged_ci_retries"]:
            reasons.append("Unchanged retry exhausted; require a diagnosed fix or changed condition")
    stalled = snapshot["sessions_without_progress"]
    if action != "research" and stalled >= policy["escalate_after_no_progress_sessions"]:
        reasons.append("Stop the stalled path; report one actionable blocker if no safe alternative exists")
    elif action != "research" and stalled >= policy["replan_after_no_progress_sessions"]:
        reasons.append("Replan the smallest testable step before further implementation")
    return {"allowed": not reasons, "reasons": reasons}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ci", "preflight"))
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        policy = json.loads((ROOT / "ops/efficiency-policy.json").read_text(encoding="utf-8"))
        validate_policy(policy)
        if args.mode == "preflight":
            require(args.snapshot is not None, "Snapshot required")
            result = preflight(json.loads(args.snapshot.read_text(encoding="utf-8")), policy)
            print(json.dumps(result))
            return 0 if result["allowed"] else 2
        workflow_dir = ROOT / ".github/workflows"
        files = set(workflow_dir.glob("*.yml")) | set(workflow_dir.glob("*.yaml"))
        require(files == {workflow_dir / "tests.yml"}, "Unreviewed extra workflow")
        workflow = json.loads((workflow_dir / "tests.yml").read_text(encoding="utf-8"))
        validate_workflow(workflow, policy)
        paths = git_changes(os.getenv("BASE_SHA", ""), os.getenv("HEAD_SHA", ""),
                            os.getenv("GITHUB_EVENT_NAME") == "pull_request")
        full = str(needs_runtime_tests(paths)).lower()
        print(json.dumps({"policy": "passed", "run_full_tests": full, "diff_verified": paths is not None}))
        if output := os.getenv("GITHUB_OUTPUT"):
            with open(output, "a", encoding="utf-8") as handle:
                handle.write("run_full_tests=" + full + "\n")
        return 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        # Do not echo untrusted snapshot/file data into public logs.
        print("Guardrail validation failed; inspect policy and input locally.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
