"""Lap performance analysis - sectors, stints, and driver comparisons."""

import pandas as pd
import numpy as np


def summarise_laps(laps):
    """Create a summary table of lap times, sectors and compounds.

    Takes raw FastF1 laps and converts timedelta columns into float
    seconds so they're easier to work with downstream.

    Returns a DataFrame with LapNumber, LapTime (seconds), Sector1/2/3
    times (seconds), Compound, TyreLife, and Stint number.
    """
    df = laps[
        ["LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
         "Compound", "TyreLife", "Stint"]
    ].copy()

    # timedeltas -> float seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        df[col] = df[col].dt.total_seconds()

    return df.reset_index(drop=True)


def sector_analysis(laps):
    """Break down average sector times by compound.

    Groups laps by tyre compound and computes the mean time through
    each sector. Sorted by overall lap time so the fastest compound
    appears first.
    """
    summary = summarise_laps(laps)
    grouped = summary.groupby("Compound")[
        ["Sector1Time", "Sector2Time", "Sector3Time", "LapTime"]
    ].mean()
    return grouped.sort_values("LapTime")


def stint_summary(laps):
    """Summarise each stint: compound, number of laps, mean/best lap time.

    Returns a DataFrame with one row per stint showing the compound used,
    how many laps were completed, average and best pace, and the tyre life
    range across that stint.
    """
    summary = summarise_laps(laps)
    stints = summary.groupby("Stint").agg(
        Compound=("Compound", "first"),
        NumLaps=("LapNumber", "count"),
        MeanLapTime=("LapTime", "mean"),
        BestLapTime=("LapTime", "min"),
        StartTyreLife=("TyreLife", "min"),
        EndTyreLife=("TyreLife", "max"),
    )
    return stints.reset_index()


def tyre_degradation(laps):
    """Calculate lap-over-lap time differences within each stint.

    Computes the delta between consecutive laps on the same set of tyres.
    Positive delta means the driver was slower than the previous lap
    (tyre dropping off). Negative means they gained time.

    Returns a DataFrame with Stint, Compound, TyreLife, LapTime,
    and LapDelta columns.
    """
    summary = summarise_laps(laps)
    summary = summary.sort_values(["Stint", "LapNumber"])
    summary["LapDelta"] = summary.groupby("Stint")["LapTime"].diff()
    return summary[["Stint", "Compound", "TyreLife", "LapTime", "LapDelta"]].dropna()


def compare_drivers(session, driver_1, driver_2):
    """Compare two drivers' lap times side by side.

    Merges both drivers' laps on LapNumber and computes the gap.
    Positive delta means driver_1 was slower than driver_2.
    """
    laps_1 = summarise_laps(session.laps.pick_drivers(driver_1).pick_accurate())
    laps_2 = summarise_laps(session.laps.pick_drivers(driver_2).pick_accurate())

    merged = pd.merge(
        laps_1[["LapNumber", "LapTime"]],
        laps_2[["LapNumber", "LapTime"]],
        on="LapNumber",
        suffixes=(f"_{driver_1}", f"_{driver_2}"),
    )
    merged["Delta"] = merged[f"LapTime_{driver_1}"] - merged[f"LapTime_{driver_2}"]
    return merged


def analyse_tyre_stints(session, driver):
    """Full tyre stint breakdown for a driver.

    Convenience function that runs stint summary, degradation, and
    sector analysis in one call.

    Returns a dict with keys: "stints", "degradation", "sectors".
    """
    laps = session.laps.pick_drivers(driver).pick_accurate()
    return {
        "stints": stint_summary(laps),
        "degradation": tyre_degradation(laps),
        "sectors": sector_analysis(laps),
    }
