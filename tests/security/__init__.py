"""Keep security tests from shadowing the application security package."""

from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
_src_security = _repo_root / "src" / "security"
_unit_security = _repo_root / "tests" / "unit" / "security"

for _path in (str(_src_security), str(_unit_security)):
    if _path not in __path__:
        __path__.append(_path)
