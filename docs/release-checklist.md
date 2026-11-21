# Source publication checklist

The project uses Apache-2.0; see [LICENSE](../LICENSE) and [third-party notices](../THIRD_PARTY.md). This checklist covers source publication. It does not require publishing wheels, hosted services or packages to PyPI.

1. Publish only the standalone source tree, with fresh repository history. Do not expose a development monorepo or import its history, credentials, calibration fixtures or CI artifacts.
2. Run `python tools/check_source_hygiene.py --tracked-only` in the checkout and scan the exact exported source tree again without that flag. Review findings without printing possible secret values. Passing pattern checks does not prove absence of unknown or deliberately obfuscated secrets.
3. Run `python tools/check_release.py` to check license and security metadata. It is a mechanical check, not independent ownership or trademark clearance. Preserve contributor notices and confirm permission for future contributed material.
4. Build from source and install into a clean environment. Run installed-package tests, CLI examples and value-only modes. Keep local build outputs out of the source archive and repository.
5. Record source hashes, actual test environment and benchmark scope. Preserve unsupported methods, cap failures and accuracy failures in results; they are not speedup claims.
6. Create the standalone repository in the intended account, push the audited commit, verify its files and visibility, and provide a verified link. A local commit alone is not publication.

Source installation uses `python -m pip install .` and needs a local C++17 compiler. Future binary releases need their own platform and bundled-dependency review. No CI job automatically publishes packages or changes repository visibility.

Return to [README](../README.md).
