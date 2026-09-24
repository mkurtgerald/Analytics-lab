"""Regression tests for efficient, fail-closed development controls."""
import os
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from tools import guardrails as g

ROOT = Path(__file__).resolve().parents[1]


class GuardrailsTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / 'ops/efficiency-policy.json').read_text())
        self.workflow = json.loads((ROOT / '.github/workflows/tests.yml').read_text())
        self.snapshot = {
            'repository': g.REPOSITORY, 'live_state_verified': True,
            'action': 'implement', 'open_implementation_prs': 1,
            'active_runs_for_head': 0, 'unchanged_ci_retries': 0,
            'ci_dispatches_this_session': 0, 'sessions_without_progress': 0,
        }

    def test_current_policy(self):
        g.validate_policy(self.policy)

    def test_current_workflow(self):
        g.validate_workflow(self.workflow, self.policy)

    def test_invalid_policy_limits(self):
        for key, maximum in g.CEILINGS.items():
            for value in [0, -1, True, '1', 1.5, maximum + 1]:
                with self.subTest(key=key, value=value):
                    policy = {**self.policy, key: value}
                    with self.assertRaises(ValueError):
                        g.validate_policy(policy)

    def test_approval_boundaries(self):
        for key in g.DENIED:
            with self.subTest(key=key), self.assertRaises(ValueError):
                g.validate_policy({**self.policy, key: True})

    def test_missing_policy_key(self):
        del self.policy['max_active_implementation_prs']
        with self.assertRaises(ValueError):
            g.validate_policy(self.policy)

    def test_repo_scope(self):
        with self.assertRaises(ValueError):
            g.validate_policy({**self.policy, 'repository': 'another/repository'})

    def test_docs_only(self):
        self.assertFalse(g.needs_runtime_tests(['README.md', 'docs/PROJECT_STATE.md', 'docs/nested/test.md']))

    def test_unknown_and_empty_diff_run_full(self):
        self.assertTrue(g.needs_runtime_tests(None))
        self.assertTrue(g.needs_runtime_tests([]))

    def test_code_and_control_changes_run_full(self):
        for path in ['analytics_lab/temporal.py', 'AGENTS.md', 'LICENSE',
                     'THIRD_PARTY.md', 'docs/script.py', 'docs/schema.json',
                     'ops/efficiency-policy.json', '.github/workflows/tests.yml',
                     'tests/test_guardrails.py', 'docs/file.MD']:
            with self.subTest(path=path):
                self.assertTrue(g.needs_runtime_tests(['README.md', path]))

    def test_suspicious_paths_run_full(self):
        for path in ['', '/docs/x.md', 'docs/../code.md', 'docs\\x.md', None]:
            with self.subTest(path=path):
                self.assertTrue(g.needs_runtime_tests([path]))

    def test_invalid_git_refs_fail_safe_without_subprocess(self):
        with patch.object(g.subprocess, 'run') as run:
            for base in ['', '0' * 40, 'main', '--help', '${bad}']:
                self.assertIsNone(g.git_changes(base, 'a' * 40, True))
            run.assert_not_called()

    def test_missing_git_history_runs_full(self):
        with patch.object(g.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'git')):
            self.assertIsNone(g.git_changes('b' * 40, 'a' * 40, True))

    def test_git_timeout_runs_full(self):
        with patch.object(g.subprocess, 'run', side_effect=subprocess.TimeoutExpired('git', 20)):
            self.assertIsNone(g.git_changes('b' * 40, 'a' * 40, True))

    def test_diff_preserves_renamed_and_deleted_paths(self):
        result = subprocess.CompletedProcess([], 0, b'docs/old.md\x00analytics_lab/new.py\x00')
        with patch.object(g.subprocess, 'run', return_value=result) as run:
            paths = g.git_changes('b' * 40, 'a' * 40, True)
            self.assertTrue(g.needs_runtime_tests(paths))
            self.assertIn('--no-renames', run.call_args.args[0])
            self.assertIn('b' * 40 + '...' + 'a' * 40, run.call_args.args[0])

    def test_push_uses_two_dot_diff_arguments(self):
        with patch.object(g.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, b'README.md\x00')) as run:
            self.assertFalse(g.needs_runtime_tests(g.git_changes('b' * 40, 'a' * 40, False)))
            self.assertEqual(run.call_args.args[0][-3:], ['b' * 40, 'a' * 40, '--'])

    def test_ci_trigger_changes_rejected(self):
        self.workflow['on']['schedule'] = [{'cron': '* * * * *'}]
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_ci_permission_changes_rejected(self):
        self.workflow['permissions']['contents'] = 'write'
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_ci_concurrency_changes_rejected(self):
        self.workflow['concurrency']['cancel-in-progress'] = False
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_ci_runner_changes_rejected(self):
        self.workflow['jobs']['linux']['runs-on'] = 'self-hosted'
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_ci_timeout_changes_rejected(self):
        self.workflow['jobs']['linux']['timeout-minutes'] = 6
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_ci_new_jobs_rejected(self):
        self.workflow['jobs']['extra'] = self.workflow['jobs']['linux']
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_unpinned_action_rejected(self):
        self.workflow['jobs']['linux']['steps'][0]['uses'] = 'actions/checkout@main'
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_persistent_credentials_rejected(self):
        self.workflow['jobs']['linux']['steps'][0]['with']['persist-credentials'] = True
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_fail_open_step_rejected(self):
        self.workflow['jobs']['linux']['steps'][2]['continue-on-error'] = True
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_fail_open_job_rejected(self):
        self.workflow['jobs']['linux']['continue-on-error'] = True
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_required_runtime_tests_cannot_be_removed(self):
        self.workflow['jobs']['windows']['steps'] = []
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_required_runtime_tests_cannot_be_skipped(self):
        self.workflow['jobs']['windows']['steps'][2]['if'] = '${{ false }}'
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_always_on_guardrails_cannot_be_removed(self):
        self.workflow['jobs']['linux']['steps'] = self.workflow['jobs']['linux']['steps'][3:]
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_windows_requires_linux(self):
        self.workflow['jobs']['windows']['needs'] = []
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_aggregate_gate_always_checks(self):
        self.workflow['jobs']['quality-gate']['if'] = '${{ success() }}'
        with self.assertRaises(ValueError):
            g.validate_workflow(self.workflow, self.policy)

    def test_preflight_allows_bounded_work(self):
        self.assertTrue(g.preflight(self.snapshot, self.policy)['allowed'])

    def test_duplicate_prs_block(self):
        self.snapshot['open_implementation_prs'] = 2
        self.assertFalse(g.preflight(self.snapshot, self.policy)['allowed'])

    def test_active_ci_blocks_mutation(self):
        self.snapshot['active_runs_for_head'] = 1
        self.assertFalse(g.preflight(self.snapshot, self.policy)['allowed'])

    def test_narrow_research_can_continue_while_ci_runs(self):
        self.snapshot.update(action="research", active_runs_for_head=1, sessions_without_progress=2)
        self.assertTrue(g.preflight(self.snapshot, self.policy)['allowed'])

    def test_first_unchanged_retry_permitted(self):
        self.snapshot['action'] = 'retry_ci'
        self.assertTrue(g.preflight(self.snapshot, self.policy)['allowed'])

    def test_second_unchanged_retry_blocked(self):
        self.snapshot.update(action='retry_ci', unchanged_ci_retries=1)
        self.assertFalse(g.preflight(self.snapshot, self.policy)['allowed'])

    def test_ci_dispatch_budget(self):
        self.snapshot.update(action='dispatch_ci', ci_dispatches_this_session=2)
        self.assertFalse(g.preflight(self.snapshot, self.policy)['allowed'])

    def test_two_stalled_sessions_require_replan(self):
        self.snapshot['sessions_without_progress'] = 2
        self.assertIn('Replan', g.preflight(self.snapshot, self.policy)['reasons'][0])

    def test_three_stalled_sessions_stop_path(self):
        self.snapshot['sessions_without_progress'] = 3
        self.assertIn('Stop', g.preflight(self.snapshot, self.policy)['reasons'][0])

    def test_unverified_snapshot_rejected(self):
        self.snapshot['live_state_verified'] = False
        with self.assertRaises(ValueError):
            g.preflight(self.snapshot, self.policy)

    def test_bad_snapshot_counts(self):
        for value in [-1, True, '0', None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                g.preflight({**self.snapshot, 'open_implementation_prs': value}, self.policy)

    def test_unknown_action_rejected(self):
        self.snapshot['action'] = 'deploy'
        with self.assertRaises(ValueError):
            g.preflight(self.snapshot, self.policy)

    @unittest.skipUnless(os.name == "posix", "The final workflow gate executes on Ubuntu only")
    def test_gate_rejects_missing_or_failed_results(self):
        # Execute the actual workflow gate, not a rewritten approximation.
        import os
        script = self.workflow['jobs']['quality-gate']['steps'][0]['run']
        for linux, windows, full, allowed in [
            ('success', 'success', 'true', True),
            ('success', 'skipped', 'false', True),
            ('failure', 'skipped', 'true', False),
            ('success', 'skipped', 'true', False),
            ('success', 'failure', 'true', False),
            ('success', 'cancelled', 'true', False),
            ('success', 'success', '', False),
        ]:
            with self.subTest(linux=linux, windows=windows, full=full):
                result = subprocess.run(['bash', '-c', script], env={**os.environ,
                    'LINUX_RESULT': linux, 'WINDOWS_RESULT': windows, 'FULL_TESTS': full},
                    capture_output=True, timeout=10)
                self.assertEqual(result.returncode == 0, allowed)


if __name__ == '__main__':
    unittest.main()
