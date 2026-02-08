"""CLI entry point — load a session and run single-driver or head-to-head analysis."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from f1_telemetry.loader import load_session, get_driver_laps, get_telemetry, list_drivers
from f1_telemetry.analysis import summarise_laps, stint_summary, tyre_degradation, compare_drivers
from f1_telemetry.modelling import fit_tyre_deg_model, fit_race_pace_model
from f1_telemetry.visualisation import (
    plot_speed_trace, plot_lap_times, plot_tyre_degradation, plot_stint_pace,
    plot_driver_comparison,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def build_parser():
    parser = argparse.ArgumentParser(description="F1 Telemetry Analysis")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--race", type=str, required=True)
    parser.add_argument("--session", type=str, default="R",
                        choices=["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"])
    parser.add_argument("--drivers", nargs="+", metavar="DRIVER")
    return parser


def _save_plots(plot_fns):
    """Run each plot function and save the figure to output/."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.ion()
    for fname, fn in tqdm(plot_fns, desc="Generating plots"):
        fn()
        plt.gcf().savefig(OUTPUT_DIR / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.ioff()
    plt.show()
    print(f"\nPlots saved to {OUTPUT_DIR}/")


def analyse_single_driver(session, driver):
    laps = get_driver_laps(session, driver)
    if laps.empty:
        print(f"No laps found for {driver}")
        return

    summary = summarise_laps(laps)
    event = f"{session.event['EventName']} {session.event.year}"

    stints = stint_summary(laps)
    print(f"\nStint summary for {driver}:")
    print(stints.to_string(index=False))

    deg_models = fit_tyre_deg_model(laps)
    print(f"\nTyre deg models:")
    for compound, info in deg_models.items():
        print(f"  {compound}: R² = {info['r2']:.3f}")

    pace = fit_race_pace_model(laps)
    print(f"\nRace pace model R² = {pace['r2']:.3f}")

    # predictions for the pace plot overlay
    lap_nums = np.arange(summary["LapNumber"].min(), summary["LapNumber"].max() + 1)
    preds = pd.DataFrame({
        "LapNumber": lap_nums,
        "PredictedTime": pace["model"].predict(lap_nums.reshape(-1, 1)),
    })

    fastest_tel = get_telemetry(laps.pick_fastest())

    _save_plots([
        ("speed_trace", lambda: plot_speed_trace(fastest_tel, title=f"{driver} Speed Trace — {event}")),
        ("lap_times", lambda: plot_lap_times(summary, title=f"{driver} Lap Times — {event}")),
        ("tyre_deg", lambda: plot_tyre_degradation(tyre_degradation(laps), title=f"{driver} Tyre Deg — {event}")),
        ("race_pace", lambda: plot_stint_pace(summary, predictions=preds, title=f"{driver} Race Pace — {event}")),
    ])


def analyse_two_drivers(session, drv1, drv2):
    event = f"{session.event['EventName']} {session.event.year}"

    for drv in (drv1, drv2):
        laps = get_driver_laps(session, drv)
        if laps.empty:
            print(f"No laps found for {drv}")
            return
        print(f"\nStint summary for {drv}:")
        print(stint_summary(laps).to_string(index=False))

    gap = compare_drivers(session, drv1, drv2)
    mean = gap["Delta"].mean()
    sign = "+" if mean > 0 else ""
    print(f"\nAverage gap: {drv1} is {sign}{mean:.3f}s vs {drv2}")

    laps_1 = get_driver_laps(session, drv1)
    laps_2 = get_driver_laps(session, drv2)
    tel_1 = get_telemetry(laps_1.pick_fastest())
    tel_2 = get_telemetry(laps_2.pick_fastest())

    _save_plots([
        ("driver_comparison", lambda: plot_driver_comparison(tel_1, tel_2, drv1, drv2, title=f"{drv1} vs {drv2} — {event}")),
        (f"{drv1.lower()}_lap_times", lambda: plot_lap_times(summarise_laps(laps_1), title=f"{drv1} Lap Times — {event}")),
        (f"{drv2.lower()}_lap_times", lambda: plot_lap_times(summarise_laps(laps_2), title=f"{drv2} Lap Times — {event}")),
    ])


def main():
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
        # default to whoever won
        results = session.results
        winner = results.loc[results["Position"] == 1.0, "Abbreviation"].iloc[0]
        print(f"No driver specified, using session winner: {winner}")
        analyse_single_driver(session, winner)


if __name__ == "__main__":
    main()