"""Direct HTTP client for JPL Horizons API.

Bypasses astroquery's timeout issues and provides robust
connection handling with retries and local caching.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

from stellar_predictor.data.models import CelestialBody, StellarSystem

# JPL Horizons IDs
HORIZONS_IDS = {
    "Sun": "10",
    "Mercury": "199",
    "Venus": "299",
    "Earth": "399",
    "Mars": "499",
    "Jupiter": "599",
    "Saturn": "699",
    "Uranus": "799",
    "Neptune": "899",
    "Pluto": "999",
}

SOLAR_SYSTEM_MASSES = {
    "Sun": 1.0,
    "Mercury": 1.66012e-7,
    "Venus": 2.44783e-6,
    "Earth": 3.00273e-6,
    "Mars": 3.22715e-7,
    "Jupiter": 9.54786e-4,
    "Saturn": 2.85837e-4,
    "Uranus": 4.36624e-5,
    "Neptune": 5.15138e-5,
    "Pluto": 7.40e-9,
}

CACHE_DIR = Path("data/cache/jpl")

# HTTP statuses worth retrying (transient server-side / throttling errors).
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

JPL_API = "https://ssd.jpl.nasa.gov/api/horizons.api"

MONTH_MAP = {i: calendar.month_abbr[i] for i in range(1, 13)}


def _to_jpl_date(iso_date: str) -> str:
    """Convert ISO date (YYYY-MM-DD) to JPL format (YYYY-Mon-DD)."""
    parts = iso_date.strip().split("-")
    if len(parts) == 3:
        try:
            month = int(parts[1])
            if 1 <= month <= 12:
                return f"{parts[0]}-{MONTH_MAP[month]}-{parts[2]}"
        except ValueError:
            pass
    return iso_date  # Already in JPL format or unknown


def _advance_date(jpl_date: str, days: int) -> str:
    """Advance a JPL-format date by N days."""
    # Try parsing JPL format "YYYY-Mon-DD"
    try:
        dt = datetime.strptime(jpl_date, "%Y-%b-%d")
        dt += timedelta(days=days)
        return f"{dt.year}-{calendar.month_abbr[dt.month]}-{dt.day:02d}"
    except ValueError:
        return jpl_date


class JPLClient:
    """Direct HTTP client for JPL Horizons ephemeris data.

    Responses are cached on disk (``cache_dir``) so repeated queries work
    offline. The cache directory is created lazily on first write, never at
    import time. Transient failures (timeouts, connection errors, HTTP
    429/5xx) are retried with exponential backoff; anything else fails fast.
    """

    def __init__(self, timeout: int = 30, max_retries: int = 3,
                 cache_dir: Path | str | None = None,
                 backoff_s: float = 1.0):
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
        self.backoff_s = backoff_s
        self.session = requests.Session()

    def _cache_key(self, *args) -> str:
        raw = "|".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _read_cache(self, cache_file: Path) -> str | None:
        try:
            text = cache_file.read_text()
        except OSError:
            return None
        # An empty/truncated file (e.g. interrupted write) is not a valid hit.
        return text if "$$SOE" in text else None

    def _write_cache(self, cache_file: Path, text: str) -> None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(text)
        except OSError:
            pass  # read-only FS etc.: caching is best-effort

    def _query(self, params: dict) -> str:
        """Query JPL Horizons API with retries and an on-disk cache."""
        cache_key = self._cache_key(json.dumps(params, sort_keys=True))
        cache_file = self.cache_dir / f"{cache_key}.txt"

        cached = self._read_cache(cache_file)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            if attempt and self.backoff_s > 0:
                time.sleep(self.backoff_s * 2 ** (attempt - 1))
            try:
                resp = self.session.get(
                    JPL_API, params=params, timeout=self.timeout
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                continue
            if resp.status_code in RETRYABLE_STATUS:
                last_error = requests.HTTPError(
                    f"HTTP {resp.status_code} from JPL Horizons", response=resp
                )
                continue
            resp.raise_for_status()  # other 4xx: a bad request, don't retry
            text = resp.text
            if "Bad dates" in text or "start must be earlier" in text:
                raise ValueError("JPL date error (try different date format)")
            if "$$SOE" in text:  # only cache responses that carry data
                self._write_cache(cache_file, text)
            return text

        raise ConnectionError(
            f"JPL Horizons unreachable after {self.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _horizons_id(body: str) -> str:
        try:
            return HORIZONS_IDS[body]
        except KeyError:
            raise ValueError(
                f"Unknown body '{body}'. Known: {', '.join(HORIZONS_IDS)}"
            ) from None

    def get_ephemeris(
        self,
        body: str,
        start_time: str,
        stop_time: str,
        step: str = "30d",
    ) -> dict:
        """Fetch ephemeris vectors for a body.

        Returns dict with keys: times_jd, positions_au (N,3), velocities_au_day (N,3)
        """
        obj_id = self._horizons_id(body)
        jpl_start = _to_jpl_date(start_time)
        jpl_stop = _to_jpl_date(stop_time)

        # JPL requires start < stop; for single-point queries, advance stop 1 day
        if jpl_start == jpl_stop:
            jpl_stop = _advance_date(jpl_stop, 1)

        params = {
            "format": "text",
            "COMMAND": f'"{obj_id}"',
            "OBJ_DATA": "YES",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "VECTORS",
            "CENTER": "@sun",
            "START_TIME": f'"{jpl_start}"',
            "STOP_TIME": f'"{jpl_stop}"',
            "STEP_SIZE": f'"{step}"',
            "VEC_TABLE": "2",  # type 2 = state vectors
            "OUT_UNITS": "AU-D",
            "REF_SYSTEM": "ICRF",
            "REF_PLANE": "ECLIPTIC",
            "CSV_FORMAT": "YES",
        }

        text = self._query(params)

        # Parse the CSV data
        lines = text.split("\n")
        in_data = False
        jd_list = []
        x_list, y_list, z_list = [], [], []
        vx_list, vy_list, vz_list = [], [], []

        for line in lines:
            if line.startswith("$$SOE"):
                in_data = True
                continue
            if line.startswith("$$EOE"):
                break
            if not in_data or line.startswith("$$") or not line.strip():
                continue

            # Parse: JDTDB, CalendarDate, X, Y, Z, VX, VY, VZ
            parts = line.strip().split(",")
            if len(parts) >= 8:
                try:
                    jd_list.append(float(parts[0]))
                    x_list.append(float(parts[2]))
                    y_list.append(float(parts[3]))
                    z_list.append(float(parts[4]))
                    vx_list.append(float(parts[5]))
                    vy_list.append(float(parts[6]))
                    vz_list.append(float(parts[7]))
                except (ValueError, IndexError):
                    continue

        if not jd_list:
            raise ValueError(f"No ephemeris data found for {body}: {text[:500]}")

        return {
            "body": body,
            "times_jd": np.array(jd_list, dtype=float),
            "positions_au": np.column_stack([
                np.array(x_list), np.array(y_list), np.array(z_list)
            ]),
            "velocities_au_day": np.column_stack([
                np.array(vx_list), np.array(vy_list), np.array(vz_list)
            ]),
        }

    def get_orbital_elements(self, body: str, epoch: str = "2000-Jan-01") -> dict:
        """Fetch osculating orbital elements at epoch."""
        obj_id = self._horizons_id(body)
        jpl_epoch = _to_jpl_date(epoch)
        params = {
            "format": "text",
            "COMMAND": f'"{obj_id}"',
            "OBJ_DATA": "YES",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "ELEMENTS",
            "CENTER": "@sun",
            "START_TIME": f'"{jpl_epoch}"',
            "STOP_TIME": f'"{jpl_epoch}"',
            "STEP_SIZE": "1d",
            "CSV_FORMAT": "YES",
        }

        text = self._query(params)

        # Parse elements
        in_data = False
        for line in text.split("\n"):
            if line.startswith("$$SOE"):
                in_data = True
                continue
            if line.startswith("$$EOE"):
                break
            if not in_data or "EC=" not in line:
                continue

            parts = line.strip().split(",")
            if len(parts) >= 10:
                return {
                    "eccentricity": float(parts[2].strip()),
                    "inclination_deg": float(parts[4].strip()),
                    "Omega_deg": float(parts[5].strip()),
                    "w_deg": float(parts[6].strip()),
                    "M_deg": float(parts[9].strip()),
                    "a_au": float(parts[11].strip()) if len(parts) > 11 else 0,
                    "period_yr": float(parts[8].strip()) if len(parts) > 8 else 0,
                }

        return {}

    def fetch_system_state(
        self,
        bodies: list[str] | None = None,
        epoch_date: str = "2000-Jan-01",
    ) -> StellarSystem:
        """Build a StellarSystem from JPL state vectors at a given epoch.

        Args:
            bodies: Bodies to include (default: Sun + all 8 planets)
            epoch_date: Date string for the state vectors (ISO or JPL format)

        Returns:
            StellarSystem with positions and velocities at the epoch
        """
        if bodies is None:
            bodies = ["Sun", "Mercury", "Venus", "Earth", "Mars",
                      "Jupiter", "Saturn", "Uranus", "Neptune"]

        system = StellarSystem(name="Solar System", source="JPL Horizons")

        for name in bodies:
            if name == "Sun":
                body = CelestialBody(
                    name="Sun", mass=1.0,
                    position=np.zeros(3), velocity=np.zeros(3),
                )
            else:
                ephemeris = self.get_ephemeris(name, epoch_date, epoch_date, "1d")
                body = CelestialBody(
                    name=name,
                    mass=SOLAR_SYSTEM_MASSES.get(name, 0.0),
                    position=ephemeris["positions_au"][0],
                    velocity=ephemeris["velocities_au_day"][0],
                    metadata={"source": "JPL Horizons", "epoch": epoch_date},
                )
            system.add_body(body)

        return system
