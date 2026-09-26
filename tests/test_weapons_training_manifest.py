import unittest

from analytics_lab.weapons_training_manifest import (
    MAX_PROOF_RUNTIME_MINUTES,
    RFDETR_LICENSE,
    RFDETR_REPOSITORY,
    RFDETR_REVISION,
    RFDETR_VARIANT,
    WEAPON_TAXONOMY,
    WeaponTrainingAsset,
    WeaponTrainingPlan,
    manifest_digest,
    validate_training_manifest,
)


def _asset(
    *,
    asset_id: str,
    split: str,
    sha: str,
    annotation_sha: str,
    source_class: str = "rifle",
    commercial_training: bool = True,
    commercial_evaluation: bool = True,
    identifiable_people: bool = False,
    third_party_rights_cleared: bool = True,
) -> WeaponTrainingAsset:
    return WeaponTrainingAsset(
        asset_id=asset_id,
        source_page_url=f"https://example.com/page/{asset_id}",
        asset_url=f"https://example.com/media/{asset_id}.jpg",
        license_id="CC0-1.0",
        rights_basis="fixture rights basis",
        source_class=source_class,
        split=split,
        expected_size=1234,
        sha256=sha,
        width=640,
        height=480,
        annotation_id=f"annotation-{asset_id}",
        annotation_sha256=annotation_sha,
        commercial_training_allowed=commercial_training,
        commercial_evaluation_allowed=commercial_evaluation,
        identifiable_people=identifiable_people,
        third_party_rights_cleared=third_party_rights_cleared,
    )


class WeaponsTrainingManifestTests(unittest.TestCase):
    def test_taxonomy_is_exact_and_mutually_exclusive(self):
        self.assertEqual(
            WEAPON_TAXONOMY,
            ("handgun", "rifle", "shotgun", "kitchen_knife", "knife", "sword", "weapon"),
        )
        self.assertEqual(len(set(WEAPON_TAXONOMY)), len(WEAPON_TAXONOMY))

    def test_fallback_plan_is_weightless_offline_cpu_bounded(self):
        plan = WeaponTrainingPlan()
        self.assertEqual(plan.architecture_repository, RFDETR_REPOSITORY)
        self.assertEqual(plan.architecture_revision, RFDETR_REVISION)
        self.assertEqual(plan.architecture_license, RFDETR_LICENSE)
        self.assertEqual(plan.model_variant, RFDETR_VARIANT)
        self.assertIsNone(plan.pretrain_weights)
        self.assertFalse(plan.network_access_allowed)
        self.assertFalse(plan.gpu_allowed)
        self.assertFalse(plan.artifact_retention_allowed)
        self.assertEqual(plan.max_proof_runtime_minutes, MAX_PROOF_RUNTIME_MINUTES)

    def test_manifest_requires_independent_train_and_validation_assets(self):
        train = _asset(
            asset_id="train-rifle",
            split="train",
            sha="1" * 64,
            annotation_sha="2" * 64,
        )
        validation = _asset(
            asset_id="validation-rifle",
            split="validation",
            sha="3" * 64,
            annotation_sha="4" * 64,
        )
        self.assertEqual(validate_training_manifest((train, validation)), (train, validation))

    def test_training_rejects_people_uncleared_rights_and_noncommercial_media(self):
        base = dict(
            asset_id="blocked",
            split="train",
            sha="5" * 64,
            annotation_sha="6" * 64,
        )
        for changes in (
            {"identifiable_people": True},
            {"third_party_rights_cleared": False},
            {"commercial_training": False},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(
                ValueError, "rights do not permit"
            ):
                blocked = _asset(**base, **changes)
                validation = _asset(
                    asset_id="validation",
                    split="validation",
                    sha="7" * 64,
                    annotation_sha="8" * 64,
                )
                validate_training_manifest((blocked, validation))

    def test_same_media_or_annotation_cannot_cross_splits(self):
        train = _asset(
            asset_id="train",
            split="train",
            sha="9" * 64,
            annotation_sha="a" * 64,
        )
        same_media = _asset(
            asset_id="validation",
            split="validation",
            sha="9" * 64,
            annotation_sha="b" * 64,
        )
        with self.assertRaisesRegex(ValueError, "same media asset"):
            validate_training_manifest((train, same_media))

        same_annotation = _asset(
            asset_id="validation-two",
            split="validation",
            sha="c" * 64,
            annotation_sha="a" * 64,
        )
        with self.assertRaisesRegex(ValueError, "duplicate annotation identity"):
            validate_training_manifest((train, same_annotation))

    def test_manifest_digest_is_deterministic_and_binds_rights(self):
        train = _asset(
            asset_id="train",
            split="train",
            sha="d" * 64,
            annotation_sha="e" * 64,
        )
        validation = _asset(
            asset_id="validation",
            split="validation",
            sha="f" * 64,
            annotation_sha="0" * 64,
        )
        first = manifest_digest((train, validation))
        second = manifest_digest((train, validation))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
