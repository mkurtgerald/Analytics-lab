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
    source_class: str | None = "rifle",
    annotation_box_count: int | None = None,
    annotation_classes: tuple[str, ...] | None = None,
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
        annotation_box_count=(
            annotation_box_count
            if annotation_box_count is not None
            else (0 if source_class is None else 1)
        ),
        annotation_classes=(
            annotation_classes
            if annotation_classes is not None
            else (() if source_class is None else (source_class,))
        ),
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

    def test_true_negative_is_not_relabelled_as_weapon(self):
        train = _asset(
            asset_id="train-rifle",
            split="train",
            sha="1" * 64,
            annotation_sha="2" * 64,
        )
        negative = _asset(
            asset_id="validation-chair",
            split="validation",
            sha="3" * 64,
            annotation_sha="4" * 64,
            source_class=None,
        )
        self.assertIsNone(negative.source_class)
        self.assertEqual(negative.annotation_box_count, 0)
        self.assertEqual(negative.annotation_classes, ())
        self.assertEqual(validate_training_manifest((train, negative)), (train, negative))

    def test_negative_rejects_boxes_or_classes(self):
        base = dict(
            asset_id="negative",
            split="validation",
            sha="5" * 64,
            annotation_sha="6" * 64,
            source_class=None,
        )
        for changes in (
            {"annotation_box_count": 1},
            {"annotation_classes": ("weapon",)},
            {"annotation_box_count": 1, "annotation_classes": ("weapon",)},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(
                ValueError, "negative asset"
            ):
                _asset(**base, **changes)

    def test_positive_requires_boxes_and_exact_class_summary(self):
        with self.assertRaisesRegex(ValueError, "at least one annotation box"):
            _asset(
                asset_id="positive-zero",
                split="train",
                sha="7" * 64,
                annotation_sha="8" * 64,
                source_class="rifle",
                annotation_box_count=0,
            )
        for classes in ((), ("weapon",), ("rifle", "weapon")):
            with self.subTest(classes=classes), self.assertRaisesRegex(
                ValueError, "class summary must match"
            ):
                _asset(
                    asset_id="positive-mismatch",
                    split="train",
                    sha="9" * 64,
                    annotation_sha="a" * 64,
                    source_class="rifle",
                    annotation_classes=classes,
                )

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

        negative = _asset(
            asset_id="validation-negative",
            split="validation",
            sha="b" * 64,
            annotation_sha="c" * 64,
            source_class=None,
        )
        self.assertNotEqual(first, manifest_digest((train, negative)))


if __name__ == "__main__":
    unittest.main()
