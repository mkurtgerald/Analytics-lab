from dataclasses import replace
import unittest

from analytics_lab.pose_adaptation_readiness import (
    LIGHTWEIGHT_OPENPOSE_PLAN,
    PretrainedArtifact,
    training_blockers,
    training_ready,
)


class PoseAdaptationReadinessTests(unittest.TestCase):
    def test_retained_trainer_lineage_is_exactly_pinned(self):
        plan = LIGHTWEIGHT_OPENPOSE_PLAN
        self.assertEqual(
            plan.trainer_repository,
            "Daniil-Osokin/lightweight-human-pose-estimation.pytorch",
        )
        self.assertEqual(plan.trainer_revision, "d23c284b09acf27a163e1febd511e7482cac25ed")
        self.assertEqual(plan.trainer_license_spdx, "Apache-2.0")
        self.assertEqual(
            [(d.name, d.version) for d in plan.dependencies],
            [
                ("torch", "0.4.1"),
                ("torchvision", "0.2.1"),
                ("pycocotools", "2.0.0"),
                ("opencv-python", "3.4.0.14"),
                ("numpy", "1.14.0"),
            ],
        )

    def test_current_plan_fails_closed_before_training(self):
        blockers = set(training_blockers())
        self.assertFalse(training_ready())
        self.assertIn("transitive_dependency_rights_unverified", blockers)
        self.assertIn("pretrained_weight_hash_missing", blockers)
        self.assertIn("pretrained_weight_license_unresolved", blockers)
        self.assertIn("training_split_empty", blockers)
        self.assertIn("validation_split_empty", blockers)
        self.assertIn("openvino_export_not_smoke_tested", blockers)

    def test_existing_holdouts_cannot_leak_into_adaptation(self):
        plan = LIGHTWEIGHT_OPENPOSE_PLAN
        for subject in (
            "GMDCSA24:Subject-2",
            "GMDCSA24:Subject-3",
            "GMDCSA24:Subject-4",
            "Figshare:SBJ_01",
            "Figshare:SBJ_10",
            "Figshare:SBJ_06",
            "Figshare:SBJ_03",
            "Figshare:SBJ_02",
            "Figshare:SBJ_09",
        ):
            self.assertIn(subject, plan.holdout_subjects)

        leaked = replace(plan, train_subjects=("GMDCSA24:Subject-2",))
        self.assertIn("train_holdout_subject_overlap", training_blockers(leaked))

    def test_resource_ceiling_is_cpu_only_and_no_spend(self):
        ceiling = LIGHTWEIGHT_OPENPOSE_PLAN.resource_ceiling
        self.assertTrue(ceiling.cpu_only)
        self.assertFalse(ceiling.paid_compute)
        self.assertEqual(ceiling.max_trials, 1)
        self.assertLessEqual(ceiling.max_cpu_threads, 2)
        self.assertLessEqual(ceiling.max_wall_minutes, 20)

    def test_all_gate_requirements_must_clear_together(self):
        plan = LIGHTWEIGHT_OPENPOSE_PLAN
        cleared = replace(
            plan,
            dependency_rights_status="cleared-commercial-and-redistribution",
            pretrained=PretrainedArtifact(
                name=plan.pretrained.name,
                source_url=plan.pretrained.source_url,
                sha256="a" * 64,
                license_spdx="Apache-2.0",
                rights_status="cleared-commercial-and-redistribution",
            ),
            export_smoke_tested=True,
            train_subjects=("ApprovedCorpus:train-subject-01",),
            validation_subjects=("ApprovedCorpus:validation-subject-01",),
        )
        self.assertEqual(training_blockers(cleared), ())
        self.assertTrue(training_ready(cleared))


if __name__ == "__main__":
    unittest.main()
