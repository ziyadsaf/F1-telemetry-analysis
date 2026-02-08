"""Lap-level analysis — sectors, stints, tyre deg, driver gaps."""

import pandas as pd


def summarise_laps(laps):
    """Pull out the columns we care about and convert timedeltas to seconds."""
    df = laps[
        ["LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
         "Compound", "TyreLife", "Stint"]
    ].copy()

    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        df[col] = df[col].dt.total_seconds()

    return df.reset_index(drop=True)


def sector_analysis(laps):
    summary = summarise_laps(laps)
    grouped = summary.groupby("Compound")[
        ["Sector1Time", "Sector2Time", "Sector3Time", "LapTime"]
    ].mean()
    return grouped.sort_values("LapTime")


def stint_summary(laps):
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
    """Lap-over-lap deltas within each stint. Positive delta = slower = more deg."""
    summary = summarise_laps(laps)
    summary = summary.sort_values(["Stint", "LapNumber"])

    # diff() within each stint so pit-stop laps don't pollute the deltas
    summary["LapDelta"] = summary.groupby("Stint")["LapTime"].diff()

    return summary[["Stint", "Compound", "TyreLife", "LapTime", "LapDelta"]].dropna()


def compare_drivers(session, drv1, drv2):
    """Per-lap delta between two drivers. Positive = drv1 slower."""
    laps_1 = summarise_laps(session.laps.pick_drivers(drv1).pick_accurate())
    laps_2 = summarise_laps(session.laps.pick_drivers(drv2).pick_accurate())

    merged = pd.merge(
        laps_1[["LapNumber", "LapTime"]],
        laps_2[["LapNumber", "LapTime"]],
        on="LapNumber",
        suffixes=(f"_{drv1}", f"_{drv2}"),
    )
    merged["Delta"] = merged[f"LapTime_{drv1}"] - merged[f"LapTime_{drv2}"]
    return merged


def analyse_tyre_stints(session, driver):
    laps = session.laps.pick_drivers(driver).pick_accurate()
    return {
        "stints": stint_summary(laps),
        "degradation": tyre_degradation(laps),
        "sectors": sector_analysis(laps),
    }