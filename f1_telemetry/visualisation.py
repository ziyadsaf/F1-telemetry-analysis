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
