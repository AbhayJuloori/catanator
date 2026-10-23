import shutil
import subprocess
from pathlib import Path

import pytest

MANIFEST = Path(__file__).resolve().parents[1] / "fast_engine" / "Cargo.toml"


def test_fast_engine_compiles_without_rust_warnings():
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is not installed")
    result = subprocess.run(
        [cargo, "check", "--manifest-path", str(MANIFEST), "--lib"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "warning:" not in result.stderr, result.stderr
