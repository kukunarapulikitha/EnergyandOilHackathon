"""EIA Open Data API v2 client — Bronze layer.

Fetches raw JSON responses from EIA and persists each payload to
`data/bronze/` with a timestamp. No cleaning, no transformation —
that happens downstream in `src/data/clean.py`.

Endpoints covered:
    - petroleum/crd/crpdn       Crude oil production by PADD
    - natural-gas/prod/sum      Natural gas marketed production by state
    - petroleum/pri/spt         WTI & Brent spot prices

Reference: https://www.eia.gov/opendata/documentation.php
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

EIA_BASE_URL = "https://api.eia.gov/v2"

PADD_REGIONS: dict[str, str] = {
    "R10": "PADD 1 (East Coast)",
    "R20": "PADD 2 (Midwest)",
    "R30": "PADD 3 (Gulf Coast)",
    "R40": "PADD 4 (Rocky Mountain)",
    "R50": "PADD 5 (West Coast)",
}

# Top natural gas producing states (~80% of U.S. marketed production)
NG_STATES: dict[str, str] = {
    "STX": "Texas",
    "SPA": "Pennsylvania",
    "SLA": "Louisiana",
    "SOK": "Oklahoma",
    "SWV": "West Virginia",
}


@dataclass
class EIAResponse:
    data: list[dict]
    source: str
    endpoint: str
    fetched_at: pd.Timestamp
    bronze_path: Path | None


class EIAClient:
    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 30,
        bronze_dir: Path | str = "data/bronze",
        log_path: Path | str = "data/ingest.log",
    ):
        self.api_key = api_key or os.getenv("EIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EIA_API_KEY not set. Register at "
                "https://www.eia.gov/opendata/register.php and add it to .env"
            )
        self.timeout = timeout
        self.bronze_dir = Path(bronze_dir)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ IO

    def _get(self, path: str, params: dict) -> dict:
        start = time.perf_counter()
        params = {**params, "api_key": self.api_key}
        url = f"{EIA_BASE_URL}/{path.lstrip('/')}"
        resp = requests.get(url, params=params, timeout=self.timeout)
        duration = time.perf_counter() - start
        self._log(path, resp.status_code, duration, len(resp.content))
        resp.raise_for_status()
        return resp.json()

    def _log(self, endpoint: str, status: int, duration: float, size: int) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"{ts}\t{endpoint}\tstatus={status}\tduration={duration:.2f}s\tbytes={size}\n"
        with open(self.log_path, "a") as f:
            f.write(line)
        print(f"  [ingest] {endpoint} → {status} in {duration:.2f}s ({size:,} bytes)")

    def _persist_bronze(self, name: str, payload: dict) -> Path:
        """Write raw API response to bronze/ with timestamped filename."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.bronze_dir / f"{name}_{ts}.json"
        path.write_text(json.dumps(payload, indent=2))
        return path

    # ------------------------------------------------------------------ endpoints

    def fetch_crude_production_by_padd(
        self, start: str = "2010-01", end: str | None = None
    ) -> EIAResponse:
        """Monthly crude oil field production by PADD region (Mbbl/d)."""
        params: dict = {
            "frequency": "monthly",
            "data[0]": "value",
            "facets[duoarea][]": list(PADD_REGIONS.keys()),
            "facets[product][]": "EPC0",
            "facets[process][]": "FPF",
            "start": start,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if end:
            params["end"] = end

        endpoint = "petroleum/crd/crpdn/data/"
        payload = self._get(endpoint, params)
        bronze = self._persist_bronze("eia_crude", payload)
        return EIAResponse(
            data=payload.get("response", {}).get("data", []),
            source="EIA API v2",
            endpoint=endpoint,
            fetched_at=pd.Timestamp.now(tz="UTC"),
            bronze_path=bronze,
        )

    def fetch_natural_gas_production(
        self, start: str = "2010-01", end: str | None = None
    ) -> EIAResponse:
        """Monthly marketed natural gas production by state (MMcf)."""
        params: dict = {
            "frequency": "monthly",
            "data[0]": "value",
            "facets[duoarea][]": list(NG_STATES.keys()),
            "facets[process][]": "VGM",
            "start": start,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if end:
            params["end"] = end

        endpoint = "natural-gas/prod/sum/data/"
        payload = self._get(endpoint, params)
        bronze = self._persist_bronze("eia_gas", payload)
        return EIAResponse(
            data=payload.get("response", {}).get("data", []),
            source="EIA API v2",
            endpoint=endpoint,
            fetched_at=pd.Timestamp.now(tz="UTC"),
            bronze_path=bronze,
        )

    def fetch_wti_spot_price(
        self, start: str = "2010-01-01", end: str | None = None
    ) -> EIAResponse:
        """Daily WTI crude spot price (USD/bbl) from Cushing, OK."""
        params: dict = {
            "frequency": "daily",
            "data[0]": "value",
            "facets[series][]": "RWTC",
            "start": start,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if end:
            params["end"] = end

        endpoint = "petroleum/pri/spt/data/"
        payload = self._get(endpoint, params)
        bronze = self._persist_bronze("wti_spot", payload)
        return EIAResponse(
            data=payload.get("response", {}).get("data", []),
            source="EIA API v2",
            endpoint=endpoint,
            fetched_at=pd.Timestamp.now(tz="UTC"),
            bronze_path=bronze,
        )
