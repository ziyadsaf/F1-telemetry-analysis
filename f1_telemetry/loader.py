"""Session and telemetry data loading via FastF1."""

from pathlib import Path

import fastf1
import pandas as pd


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
    """Enable FastF1 data caching. Creates the directory if it doesn't exist."""
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))


def load_session(year, race, session="R", laps=True, telemetry=True, weather=True):
    """Load an F1 session with lap, telemetry and weather data.

    Args:
        year: Season year (2018 onwards).
        race: Grand Prix name (e.g. "Bahrain") or round number.
        session: Session identifier - one of FP1, FP2, FP3, Q, SQ, S, R.
        laps: Load lap-level data.
        telemetry: Load car telemetry data.
        weather: Load weather data.

    Returns:
        A loaded FastF1 Session object.
    """
    enable_cache()
    f1_session = fastf1.get_session(year, race, session)
    f1_session.load(laps=laps, telemetry=telemetry, weather=weather, messages=False)
    return f1_session


def get_driver_laps(session, driver, accurate_only=True):
    """Get all laps for a specific driver.

    Args:
        session: A loaded Session object.
        driver: Driver abbreviation (e.g. "VER", "HAM").
        accurate_only: If True, filter out laps without accurate timing.

    Returns:
        Laps DataFrame for the driver.
    """
    laps = session.laps.pick_drivers(driver)
    if accurate_only:
        laps = laps.pick_accurate()
    return laps


def get_telemetry(lap):
    """Get merged car + position telemetry for a single lap.

    Returns a DataFrame with columns including Speed, Throttle, Brake,
    nGear, RPM, DRS, Distance, X, Y, Z.
    """
    return lap.get_telemetry()


def get_session_results(session):
    """Get session results (driver positions, teams, times)."""
    return session.results


def get_weather(session):
    """Get weather data recorded during the session."""
    return session.weather_data


def list_drivers(session):
    """Return a list of driver abbreviations that participated in the session."""
    return list(session.laps["Driver"].unique())
