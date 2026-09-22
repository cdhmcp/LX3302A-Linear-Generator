"""Static contract checks for the internal Windows release workflow."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).parent


class ReleaseAutomationTests(unittest.TestCase):
    def test_release_script_enforces_the_internal_release_contract(self):
        script = (ROOT / "scripts" / "release_windows.ps1").read_text(encoding="utf-8")
        for required_fragment in (
            "ValidatePattern('^\\d+\\.\\d+\\.\\d+$')",
            "Requested version $Version does not match pyproject.toml version",
            "status', '--porcelain', '--untracked-files=no",
            "HEAD must have the annotated tag $Tag",
            "must be an annotated Git tag",
            "win64 release archive must be built with a 64-bit",
            "python -m unittest discover -v",
            "build_windows.ps1') -SkipDependencyInstall",
            "dist\\release-staging",
            "Move-Item -LiteralPath $StagingRoot -Destination $ReleaseRoot",
            "RELEASE_MANIFEST.json",
            "INSTALL.txt",
            "Compress-Archive",
            "Get-FileHash",
            "SHA256SUMS.txt",
        ):
            with self.subTest(fragment=required_fragment):
                self.assertIn(required_fragment, script)

    def test_pyproject_version_has_a_matching_semver_release_tag_shape(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"$', pyproject, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertRegex(f"v{match.group(1)}", r"^v\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
