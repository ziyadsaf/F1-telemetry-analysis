"""Matplotlib plots for telemetry and race analysis."""

import matplotlib.pyplot as plt

# Colours matched to the F1 TV broadcast graphics
COMPOUND_COLOURS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFC906",
    "HARD": "#EEEEEE",
    "INTERMEDIATE": "#39B54A",
    "WET": "#00BFFF",
}


def _compound_colour(compound):
    return COMPOUND_COLOURS.get(compound, "#888888")


def plot_speed_trace(telemetry, title="Speed Trace"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    dist = telemetry["Distance"]

    axes[0].plot(dist, telemetry["Speed"], color="#1E88E5", linewidth=1.2)
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dist, telemetry["Throttle"], color="#43A047", linewidth=1.2)
    axes[1].set_ylabel("Throttle (%)")
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True, alpha=0.3)

    axes[2].fill_between(
        dist, telemetry["Brake"].astype(float), color="#E53935", alpha=0.7
    )
    axes[2].set_ylabel("Brake")
    axes[2].set_xlabel("Distance (m)")
    axes[2].set_ylim(-0.05, 1.1)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_lap_times(laps_summary, title="Lap Times"):
    fig, ax = plt.subplots(figsize=(14, 6))

    for compound in laps_summary["Compound"].unique():
        mask = laps_summary["Compound"] == compound
        ax.scatter(
            laps_summary.loc[mask, "LapNumber"],
            laps_summary.loc[mask, "LapTime"],
            c=_compound_colour(compound),
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
    fig, ax = plt.subplots(figsize=(14, 6))

    for (stint, compound), grp in deg_data.groupby(["Stint", "Compound"]):
        ax.plot(
            grp["TyreLife"], grp["LapTime"],
            marker="o", markersize=4,
            color=_compound_colour(compound),
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


def plot_driver_comparison(tel_1, tel_2, drv1, drv2,
                           title="Driver Comparison"):
    """Side-by-side telemetry overlay for two drivers' fastest laps."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].plot(tel_1["Distance"], tel_1["Speed"],
                 label=drv1, color="#1E88E5", linewidth=1.2)
    axes[0].plot(tel_2["Distance"], tel_2["Speed"],
                 label=drv2, color="#E53935", linewidth=1.2)
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tel_1["Distance"], tel_1["Throttle"],
                 color="#1E88E5", linewidth=1.2)
    axes[1].plot(tel_2["Distance"], tel_2["Throttle"],
                 color="#E53935", linewidth=1.2)
    axes[1].set_ylabel("Throttle (%)")
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(tel_1["Distance"], tel_1["nGear"],
                 color="#1E88E5", linewidth=1.2)
    axes[2].plot(tel_2["Distance"], tel_2["nGear"],
                 color="#E53935", linewidth=1.2)
    axes[2].set_ylabel("Gear")
    axes[2].set_xlabel("Distance (m)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_stint_pace(laps_summary, predictions=None, title="Race Pace"):
    """Lap times coloured by compound, with optional model overlay."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for compound, grp in laps_summary.groupby("Compound"):
        ax.scatter(grp["LapNumber"], grp["LapTime"],
                   color=_compound_colour(compound), edgecolors="black",
                   linewidths=0.5, s=40, label=compound, zorder=3)

    if predictions is not None:
        ax.plot(
            predictions["LapNumber"], predictions["PredictedTime"],
            color="black", linestyle="--", linewidth=1.5,
            label="Model fit", zorder=4,
        )

    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(title="Compound")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
