"""Explicit r1301 developer entry using the immutable shared Driver."""
import sys
from pathlib import Path
import capacity_scope as scope

sys.path.insert(0, str(scope.COMMON))
from runtime_driver import Driver

driver = Driver(scope, Path(__file__).resolve())
if __name__ == "__main__":
    raise SystemExit(driver.main())
