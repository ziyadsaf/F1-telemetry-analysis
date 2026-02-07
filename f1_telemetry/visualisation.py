"""Plotting and chart generation for F1 telemetry data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# F1 broadcast tyre colours
COMPOUND_COLOURS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFC906",
    "HARD": "#EEEEEE",
    "INTERMEDIATE": "#39B54A",
    "WET": "#00BFFF",
}


def plot_speed_trace(telemetry, title="Speed Trace"):
    """Plot speed, throttle and brake traces for a single lap.

    Three vertically stacked panels sharing the same x-axis (distance
    around the lap). Speed as a line, throttle as a line, brake as a
    filled area since it's essentially binary (on/off).

    Args:
        telemetry: Telemetry DataFrame from loader.get_telemetry().
        title: Plot title.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")
 
    distance = telemetry["Distance"]

    # speed
    axes[0].plot(distance, telemetry["Speed"], color="#1E88E5", linewidth=1.2)
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].grid(True, alpha=0.3)

    # throttle
    axes[1].plot(distance, telemetry["Throttle"], color="#43A047", linewidth=1.2)
    axes[1].set_ylabel("Throttle (%)")
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True, alpha=0.3)

    # brake
    axes[2].fill_between(
        distance, telemetry["Brake"].astype(float), color="#E53935", alpha=0.7
    )
    axes[2].set_ylabel("Brake")
    axes[2].set_xlabel("Distance (m)")
    axes[2].set_ylim(-0.05, 1.1)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_lap_times(laps_summary, title="Lap Times"):
    """Scatter plot of lap times coloured by tyre compound.

    Each dot is one lap. Colour maps to compound using F1 broadcast
    colours. You can spot stint transitions (colour changes) and tyre
    drop-off (dots drifting upward within a stint).

    Args:
        laps_summary: DataFrame from analysis.summarise_laps().
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for compound in laps_summary["Compound"].unique():
        mask = laps_summary["Compound"] == compound
        colour = COMPOUND_COLOURS.get(compound, "#888888")
        ax.scatter(
            laps_summary.loc[mask, "LapNumber"],
            laps_summary.loc[mask, "LapTime"],
            c=colour,
            edgecolors="black",
            linewidths=0.5,
            s=50,
            label=compound,
            zorder=3,
        )

    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(title="Compound")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_tyre_degradation(deg_data, title="Tyre Degradation"):
    """Plot lap time vs tyre life, one line per stint.

    Each stint gets its own line coloured by compound. Steeper upward
    slope means the compound is falling off faster. Useful for comparing
    degradation rates between soft, medium and hard.

    Args:
        deg_data: DataFrame from analysis.tyre_degradation().
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for (stint, compound), group in deg_data.groupby(["Stint", "Compound"]):
        colour = COMPOUND_COLOURS.get(compound, "#888888")
        ax.plot(
            group["TyreLife"],
            group["LapTime"],
            marker="o",
            markersize=4,
            color=colour,
            label=f"Stint {stint} ({compound})",
            linewidth=1.5,
        )

    ax.set_xlabel("Tyre Life (laps)")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_driver_comparison(telemetry_1, telemetry_2, driver_1, driver_2,
                           title="Driver Comparison"):
    """Overlay two drivers' telemetry on the same chart.

    Three panels: speed (top), throttle (middle), gear (bottom).
    Driver 1 in blue, driver 2 in red. Shows where one driver gains
    or loses time - higher speed through a corner, earlier throttle
    application, different gear choices.

    Args:
        telemetry_1: Telemetry DataFrame for driver 1.
        telemetry_2: Telemetry DataFrame for driver 2.
        driver_1: Driver 1 abbreviation (e.g. "VER").
        driver_2: Driver 2 abbreviation (e.g. "HAM").
        title: Plot title.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # speed
    axes[0].plot(
        telemetry_1["Distance"], telemetry_1["Speed"],
        label=driver_1, color="#1E88E5", linewidth=1.2,
    )
    axes[0].plot(
        telemetry_2["Distance"], telemetry_2["Speed"],
        label=driver_2, color="#E53935", linewidth=1.2,
    )
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # throttle
    axes[1].plot(
        telemetry_1["Distance"], telemetry_1["Throttle"],
        color="#1E88E5", linewidth=1.2,
    )
    axes[1].plot(
        telemetry_2["Distance"], telemetry_2["Throttle"],
        color="#E53935", linewidth=1.2,
    )
    axes[1].set_ylabel("Throttle (%)")
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True, alpha=0.3)

    # gear
    axes[2].plot(
        telemetry_1["Distance"], telemetry_1["nGear"],
        color="#1E88E5", linewidth=1.2,
    )
    axes[2].plot(
        telemetry_2["Distance"], telemetry_2["nGear"],
        color="#E53935", linewidth=1.2,
    )
    axes[2].set_ylabel("Gear")
    axes[2].set_xlabel("Distance (m)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_stint_pace(laps_summary, model_predictions=None, title="Race Pace"):
    """Scatter plot of race pace with optional model fit overlay.

    Same compound-coloured scatter as plot_lap_times, but designed
    to be used with the race pace model. Pass in model predictions
    to see the fitted curve as a dashed black line on top of the
    actual lap times.

    Args:
        laps_summary: DataFrame from analysis.summarise_laps().
        model_predictions: Optional DataFrame with LapNumber and
            PredictedTime columns from the race pace model.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for compound in laps_summary["Compound"].unique():
        mask = laps_summary["Compound"] == compound
        colour = COMPOUND_COLOURS.get(compound, "#888888")
        ax.scatter(
            laps_summary.loc[mask, "LapNumber"],
            laps_summary.loc[mask, "LapTime"],
            c=colour,
            edgecolors="black",
            linewidths=0.5,
            s=40,
            label=compound,
            zorder=3,
        )

    if model_predictions is not None:
        ax.plot(
            model_predictions["LapNumber"],
            model_predictions["PredictedTime"],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Model fit",
            zorder=4,
        )

    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(title="Compound")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
