"""CLI entry point for F1 telemetry analysis."""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

from f1_telemetry.loader import load_session, get_driver_laps, get_telemetry, list_drivers
from f1_telemetry.analysis import summarise_laps, stint_summary, tyre_degradation, compare_drivers
from f1_telemetry.modelling import fit_tyre_deg_model, fit_race_pace_model
from f1_telemetry.visualisation import (
    plot_speed_trace, plot_lap_times, plot_tyre_degradation, plot_stint_pace,
    plot_driver_comparison,
)




def build_parser():
    """Set up the argument parser with year, race, session and driver options."""
    parser = argparse.ArgumentParser(description="F1 Telemetry Analysis")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--race", type=str, required=True)
    parser.add_argument("--session", type=str, default="R",
                        choices=["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"])
    parser.add_argument("--drivers", nargs="+", metavar="DRIVER")
    return parser


def analyse_single_driver(session, driver):
    """Run the full analysis pipeline for a single driver."""
    laps = get_driver_laps(session, driver)
    if laps.empty:
        print(f"No laps found for {driver}")
        return

    summary = summarise_laps(laps)
    event_name = f"{session.event['EventName']} {session.event.year}"

    # print stint breakdown
    stints = stint_summary(laps)
    print(f"\nStint summary for {driver}:")
    print(stints.to_string(index=False))

    # fit tyre deg model per compound
    deg_models = fit_tyre_deg_model(laps)
    print(f"\nTyre deg models:")
    for compound, info in deg_models.items():
        print(f"  {compound}: R^2 = {info['r2']:.3f}")

    # fit race pace model
    pace = fit_race_pace_model(laps)
    print(f"\nRace pace model R^2 = {pace['r2']:.3f}")

    # build predictions for the pace overlay
    lap_nums = np.arange(summary["LapNumber"].min(), summary["LapNumber"].max() + 1)
    predictions = pd.DataFrame({
        "LapNumber": lap_nums,
        "PredictedTime": pace["model"].predict(lap_nums.reshape(-1, 1)),
    })

    # plots - interactive mode so they all open at once
    fastest = laps.pick_fastest()
    telemetry = get_telemetry(fastest)

    plt.ion()
    plots = [
        ("speed_trace", lambda: plot_speed_trace(telemetry, title=f"{driver} Speed Trace - {event_name}")),
        ("lap_times", lambda: plot_lap_times(summary, title=f"{driver} Lap Times - {event_name}")),
        ("tyre_deg", lambda: plot_tyre_degradation(tyre_degradation(laps), title=f"{driver} Tyre Deg - {event_name}")),
        ("race_pace", lambda: plot_stint_pace(summary, model_predictions=predictions, title=f"{driver} Race Pace - {event_name}")),
    ]

    OUTPUT_DIR.mkdir(exist_ok=True)
    for filename, plot_fn in tqdm(plots, desc="Generating plots"):
        plot_fn()
        plt.gcf().savefig(OUTPUT_DIR / f"{filename}.png", dpi=150, bbox_inches="tight")

    plt.ioff()
    plt.show()
    print(f"\nPlots saved to {OUTPUT_DIR}/")


def analyse_two_drivers(session, driver_1, driver_2):
    """Compare two drivers' performance and telemetry side by side."""
    event_name = f"{session.event['EventName']} {session.event.year}"

    # stint breakdowns for both
    for driver in (driver_1, driver_2):
        laps = get_driver_laps(session, driver)
        if laps.empty:
            print(f"No laps found for {driver}")
            return
        stints = stint_summary(laps)
        print(f"\nStint summary for {driver}:")
        print(stints.to_string(index=False))

    # lap time gap
    comparison = compare_drivers(session, driver_1, driver_2)
    mean_gap = comparison["Delta"].mean()
    sign = "+" if mean_gap > 0 else ""
    print(f"\nAverage gap: {driver_1} is {sign}{mean_gap:.3f}s vs {driver_2}")

    # telemetry comparison on fastest laps
    laps_1 = get_driver_laps(session, driver_1)
    laps_2 = get_driver_laps(session, driver_2)

    tel_1 = get_telemetry(laps_1.pick_fastest())
    tel_2 = get_telemetry(laps_2.pick_fastest())

    plt.ion()
    plots = [
        ("driver_comparison", lambda: plot_driver_comparison(tel_1, tel_2, driver_1, driver_2, title=f"{driver_1} vs {driver_2} - {event_name}")),
        (f"{driver_1.lower()}_lap_times", lambda: plot_lap_times(summarise_laps(laps_1), title=f"{driver_1} Lap Times - {event_name}")),
        (f"{driver_2.lower()}_lap_times", lambda: plot_lap_times(summarise_laps(laps_2), title=f"{driver_2} Lap Times - {event_name}")),
    ]

    OUTPUT_DIR.mkdir(exist_ok=True)
    for filename, plot_fn in tqdm(plots, desc="Generating plots"):
        plot_fn()
        plt.gcf().savefig(OUTPUT_DIR / f"{filename}.png", dpi=150, bbox_inches="tight")

    plt.ioff()
    plt.show()
    print(f"\nPlots saved to {OUTPUT_DIR}/")


def main():
    """Load an F1 session and run the appropriate analysis based on args."""
    args = build_parser().parse_args()

    pbar = tqdm(total=2, desc="Loading session data")
    session = load_session(args.year, args.race, args.session)
    pbar.update(1)
    pbar.set_description("Processing drivers")
    pbar.update(1)
    pbar.close()

    available = list_drivers(session)
    print(f"Drivers: {', '.join(available)}")

    if args.drivers:
        # check the drivers actually exist in this session
        for d in args.drivers:
            if d not in available:
                print(f"Error: '{d}' not found in session")
                sys.exit(1)

        if len(args.drivers) == 1:
            analyse_single_driver(session, args.drivers[0])
        elif len(args.drivers) == 2:
            analyse_two_drivers(session, args.drivers[0], args.drivers[1])
        else:
            print("Error: pass one or two drivers max")
            sys.exit(1)
    else:
        # no driver specified, just use whoever won
        results = session.results
        winner = results.loc[results["Position"] == 1.0, "Abbreviation"].iloc[0]
        print(f"No driver specified, using session winner: {winner}")
        analyse_single_driver(session, winner)


if __name__ == "__main__":
    main()
