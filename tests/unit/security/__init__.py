"""Keep unit security tests from shadowing the application security package."""

from pathlib import Path

_repo_root = Path(__file__).resolve().parents[3]
_src_security = _repo_root / "src" / "security"
_security_tests = _repo_root / "tests" / "security"

for _path in (str(_src_security), str(_security_tests)):
    if _path not in __path__:
        __path__.append(_path)
