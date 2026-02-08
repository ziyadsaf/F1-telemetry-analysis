"""Polynomial fits for tyre deg and race pace prediction."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from f1_telemetry.analysis import summarise_laps


def fit_tyre_deg_model(laps, degree=2):
    """Fit lap_time ~ poly(tyre_life) per compound.
    Degree 2 works well — deg is roughly quadratic in most cases."""
    summary = summarise_laps(laps)
    summary = summary.dropna(subset=["LapTime", "TyreLife"])

    models = {}
    for compound, grp in summary.groupby("Compound"):
        if len(grp) < 3:
            continue

        X = grp[["TyreLife"]].values
        y = grp["LapTime"].values

        pipe = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        pipe.fit(X, y)

        models[compound] = {
            "model": pipe,
            "r2": pipe.score(X, y),
            "coefficients": pipe.named_steps["linearregression"].coef_,
        }

    return models


def predict_lap_time(models, tyre_life, compound):
    if compound not in models:
        return None
    return float(models[compound]["model"].predict([[tyre_life]])[0])


def fit_race_pace_model(laps, degree=3):
    """Cubic fit to lap_time ~ lap_number. Captures the fuel-burn trend
    plus the tyre cliff without overfitting on most races."""
    summary = summarise_laps(laps)
    summary = summary.dropna(subset=["LapTime"])

    X = summary[["LapNumber"]].values
    y = summary["LapTime"].values

    pipe = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    pipe.fit(X, y)

    return {
        "model": pipe,
        "r2": pipe.score(X, y),
        "coefficients": pipe.named_steps["linearregression"].coef_,
    }


def estimate_optimal_stint_length(laps, compound, threshold=1.0):
    """Find how many laps before deg costs more than `threshold` seconds
    relative to lap 1 on that compound."""
    models = fit_tyre_deg_model(laps)
    if compound not in models:
        return None

    pipe = models[compound]["model"]
    baseline = float(pipe.predict([[1]])[0])

    for life in range(2, 60):
        if float(pipe.predict([[life]])[0]) - baseline > threshold:
            return life - 1

    return 59  # basically no deg on this compound


def fuel_corrected_pace(laps, fuel_effect=0.055):
    """Strip out the fuel-mass effect from lap times.

    The 0.055 s/lap figure comes from the ~100 kg fuel load burned roughly
    linearly over a ~57-lap race, with each kg worth about ~0.03 s at most
    circuits. It's a ballpark — some tracks are more sensitive than others."""
    summary = summarise_laps(laps)
    summary = summary.dropna(subset=["LapTime"])
    total_laps = summary["LapNumber"].max()

    # heavier car at the start = slower, so we ADD time to later laps
    # to normalise everything to "full fuel" pace
    summary["FuelCorrectedTime"] = (
        summary["LapTime"]
        + (total_laps - summary["LapNumber"]) * fuel_effect
    )
    return summary[["LapNumber", "LapTime", "FuelCorrectedTime", "Compound"]]
