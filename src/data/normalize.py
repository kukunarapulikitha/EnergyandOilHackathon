"""Back-compat shim — prefer `src.data.clean` in new code.

The Silver layer lives in `clean.py`. This module re-exports the two
legacy names (`normalize_production`, `aggregate_annual`) so older
callers keep working during the refactor. Remove once all callers
migrate.
"""

from __future__ import annotations

from .clean import aggregate_annual  # re-exported
from .clean import clean_crude_production as normalize_production  # alias

__all__ = ["normalize_production", "aggregate_annual"]
