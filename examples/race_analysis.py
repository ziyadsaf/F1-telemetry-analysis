"""Example: full race analysis for the 2021 Brazilian Grand Prix."""

import numpy as np
import pandas as pd

from f1_telemetry.loader import load_session, get_driver_laps, get_telemetry
from f1_telemetry.analysis import summarise_laps, stint_summary, tyre_degradation, compare_drivers
from f1_telemetry.modelling import fit_tyre_deg_model, fit_race_pace_model
from f1_telemetry.visualisation import (
    plot_speed_trace, plot_lap_times, plot_tyre_degradation, plot_stint_pace,
    plot_driver_comparison,
)


# load the race
session = load_session(2021, "Brazilian Grand Prix", "R")

# --- Verstappen single-driver analysis ---

ver_laps = get_driver_laps(session, "VER")
ver_summary = summarise_laps(ver_laps)

# stint breakdown
stints = stint_summary(ver_laps)
print("VER stint summary:")
print(stints.to_string(index=False))

# tyre deg models
deg_models = fit_tyre_deg_model(ver_laps)
print("\nVER tyre deg models:")
for compound, info in deg_models.items():
    print(f"  {compound}: R² = {info['r2']:.3f}")

# race pace model
pace = fit_race_pace_model(ver_laps)
print(f"\nVER race pace model R² = {pace['r2']:.3f}")

# pace predictions for overlay
lap_nums = np.arange(ver_summary["LapNumber"].min(), ver_summary["LapNumber"].max() + 1)
predictions = pd.DataFrame({
    "LapNumber": lap_nums,
    "PredictedTime": pace["model"].predict(lap_nums.reshape(-1, 1)),
})

# plots
ver_fastest = ver_laps.pick_fastest()
ver_tel = get_telemetry(ver_fastest)

plot_speed_trace(ver_tel, title="VER Speed Trace - Interlagos 2021")
plot_lap_times(ver_summary, title="VER Lap Times - Interlagos 2021")
plot_tyre_degradation(tyre_degradation(ver_laps), title="VER Tyre Deg - Interlagos 2021")
plot_stint_pace(ver_summary, model_predictions=predictions, title="VER Race Pace - Interlagos 2021")

# --- Hamilton vs Verstappen comparison ---

ham_laps = get_driver_laps(session, "HAM")

comparison = compare_drivers(session, "HAM", "VER")
mean_gap = comparison["Delta"].mean()
sign = "+" if mean_gap > 0 else ""
print(f"\nAverage gap: HAM is {sign}{mean_gap:.3f}s vs VER")

# telemetry overlay on fastest laps
ham_tel = get_telemetry(ham_laps.pick_fastest())
plot_driver_comparison(ham_tel, ver_tel, "HAM", "VER", title="HAM vs VER - Interlagos 2021")

# lap times side by side
plot_lap_times(summarise_laps(ham_laps), title="HAM Lap Times - Interlagos 2021")
plot_lap_times(ver_summary, title="VER Lap Times - Interlagos 2021")