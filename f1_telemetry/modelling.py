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
