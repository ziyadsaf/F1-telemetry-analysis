"""Wrappers around FastF1 to keep the caching/loading logic in one place."""

from pathlib import Path
import fastf1


CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

VALID_SESSIONS = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "Q": "Qualifying",
    "S": "Sprint",
    "SQ": "Sprint Qualifying",
    "R": "Race",
}


def enable_cache(cache_dir=None):
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))


def load_session(year, race, session="R", laps=True, telemetry=True, weather=True):
    """Load a full session. Telemetry download can be slow the first time
    (~30-60s depending on session length) but gets cached after that."""
    enable_cache()
    f1_session = fastf1.get_session(year, race, session)
    # messages=False suppresses the live timing spam
    f1_session.load(laps=laps, telemetry=telemetry, weather=weather, messages=False)
    return f1_session


def get_driver_laps(session, driver, accurate_only=True):
    laps = session.laps.pick_drivers(driver)
    if accurate_only:
        laps = laps.pick_accurate()
    return laps


def get_telemetry(lap):
    """Merged car + position telemetry for a single lap."""
    return lap.get_telemetry()


def get_session_results(session):
    return session.results


def get_weather(session):
    return session.weather_data


def list_drivers(session):
    return list(session.laps["Driver"].unique())