from __future__ import annotations

import types

import scripts.check_environment as check_environment


def test_missing_optional_cli_tools_do_not_fail_environment_check(monkeypatch) -> None:
    """Optional agent CLIs should be informational, not fatal in CI or other non-codespace environments."""
    monkeypatch.setattr(check_environment.shutil, "which", lambda name: None)

    for package in check_environment.PACKAGES:
        monkeypatch.setitem(
            check_environment.sys.modules,
            package,
            types.SimpleNamespace(__version__="test-version"),
        )

    assert check_environment.main() == 0
