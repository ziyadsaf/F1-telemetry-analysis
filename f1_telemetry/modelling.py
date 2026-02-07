"""Lap time prediction and tyre degradation modelling."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from f1_telemetry.analysis import summarise_laps


def fit_tyre_deg_model(laps, degree=2):
    """Fit a polynomial regression to tyre degradation per compound.

    Models LapTime as a function of TyreLife for each compound in the
    data. Uses a degree-2 polynomial by default because tyre deg isn't
    linear - there's usually a "cliff" where performance drops off.

    Skips any compound with fewer than 3 laps since you can't fit
    a meaningful curve to that.

    Args:
        laps: A Laps object (from FastF1) for a single driver.
        degree: Polynomial degree for the regression.

    Returns:
        Dict mapping compound name to a dict with:
            - "model": the fitted sklearn pipeline
            - "r2": R-squared score on training data
            - "coefficients": polynomial coefficients
    """
    summary = summarise_laps(laps)
    summary = summary.dropna(subset=["LapTime", "TyreLife"])

    models = {}
    for compound, group in summary.groupby("Compound"):
        if len(group) < 3:
            continue

        X = group[["TyreLife"]].values
        y = group["LapTime"].values

        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)

        r2 = model.score(X, y)
        coeffs = model.named_steps["linearregression"].coef_

        models[compound] = {
            "model": model,
            "r2": r2,
            "coefficients": coeffs,
        }

    return models


def predict_lap_time(model, tyre_life, compound):
    """Predict lap time for a given tyre age and compound.

    Uses the output of fit_tyre_deg_model() to predict what a driver's
    lap time would be at a specific point in a stint.

    Args:
        model: Output of fit_tyre_deg_model().
        tyre_life: Number of laps on the current set of tyres.
        compound: Tyre compound name (e.g. "SOFT", "MEDIUM").

    Returns:
        Predicted lap time in seconds, or None if compound not in model.
    """
    if compound not in model:
        return None
    return float(model[compound]["model"].predict([[tyre_life]])[0])


def fit_race_pace_model(laps, degree=3):
    """Fit a polynomial model to overall race pace.

    Models lap time as a function of lap number across the full race.
    Default degree 3 captures the combined shape of fuel burn (car gets
    lighter and faster) and tyre degradation (car gets slower) plus
    the reset effect of pit stops.

    The R-squared value tells you how predictable the driver's pace was.
    A high R2 means consistent, clean race. Low R2 usually means
    incidents, safety cars, or weather changes.

    Args:
        laps: A Laps object for a single driver.
        degree: Polynomial degree (default 3).

    Returns:
        Dict with "model", "r2", and "coefficients".
    """
    summary = summarise_laps(laps)
    summary = summary.dropna(subset=["LapTime"])

    X = summary[["LapNumber"]].values
    y = summary["LapTime"].values

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)

    return {
        "model": model,
        "r2": model.score(X, y),
        "coefficients": model.named_steps["linearregression"].coef_,
    }


def estimate_optimal_stint_length(laps, compound, threshold=1.0):
    """Estimate how many laps a compound can run before deg gets too high.

    Fits the tyre deg model, then walks forward from tyre life 1 (fresh
    tyres) until the predicted lap time exceeds the fresh-tyre prediction
    by more than the threshold. That's the point where you're losing
    too much time and should pit.

    Args:
        laps: Laps object for a single driver.
        compound: Compound to evaluate (e.g. "SOFT").
        threshold: Max acceptable degradation in seconds (default 1.0).

    Returns:
        Estimated optimal stint length in laps, or None if there isn't
        enough data for that compound.
    """
    models = fit_tyre_deg_model(laps)
    if compound not in models:
        return None

    pipe = models[compound]["model"]
    baseline = float(pipe.predict([[1]])[0])

    for tyre_life in range(2, 60):
        predicted = float(pipe.predict([[tyre_life]])[0])
        if predicted - baseline > threshold:
            return tyre_life - 1

    # compound didn't degrade past threshold within 59 laps
    return 59
