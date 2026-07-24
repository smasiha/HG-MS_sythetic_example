from __future__ import annotations

import unittest
from collections import Counter

from synthetic_ula_ridge_cg.configuration import PACKAGE_ROOT
from synthetic_ula_ridge_cg.public_artifact import (
    assert_sanitization_is_lossless,
    sanitize_json,
    validate_scientific_configurations,
)


class PublicArtifactTest(unittest.TestCase):
    def test_sanitizer_removes_only_environment_metadata(self) -> None:
        source = {
            "host": "private-node",
            "cpu_count_visible": 10,
            "selected_config_path": "private-config-location",
            "seed": 1000,
            "method_config": {"branches": 16},
            "trajectory": [
                {
                    "elapsed_seconds": 0.25,
                    "hyperobjective_gap": 0.1,
                }
            ],
        }
        removed: Counter[str] = Counter()
        sanitized = sanitize_json(source, removed)
        self.assertEqual(
            sanitized,
            {
                "seed": 1000,
                "method_config": {"branches": 16},
                "trajectory": [
                    {
                        "elapsed_seconds": 0.25,
                        "hyperobjective_gap": 0.1,
                    }
                ],
            },
        )
        self.assertEqual(
            removed,
            Counter(
                {
                    "host": 1,
                    "cpu_count_visible": 1,
                    "selected_config_path": 1,
                }
            ),
        )
        assert_sanitization_is_lossless(source, sanitized)

    def test_sanitizer_rejects_identifying_scientific_strings(self) -> None:
        with self.assertRaisesRegex(ValueError, "identifying string"):
            private_path = "/" + "home/private/result.json"
            sanitize_json({"scientific_note": private_path})

    def test_frozen_scientific_protocol(self) -> None:
        validate_scientific_configurations(PACKAGE_ROOT)


if __name__ == "__main__":
    unittest.main()
