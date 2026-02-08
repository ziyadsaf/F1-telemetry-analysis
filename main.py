"""CLI entry point for F1 telemetry analysis."""

import argparse
import sys
import numpy as np
import pandas as pd

from f1_telemetry.loader import load_session, get_driver_laps, get_telemetry, list_drivers
from f1_telemetry.analysis import summarise_laps, stint_summary, tyre_degradation
from f1_telemetry.modelling import fit_tyre_deg_model, fit_race_pace_model
from f1_telemetry.visualisation import (
    plot_speed_trace, plot_lap_times, plot_tyre_degradation, plot_stint_pace,
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
        print(f"  {compound}: R² = {info['r2']:.3f}")

    # fit race pace model
    pace = fit_race_pace_model(laps)
    print(f"\nRace pace model R² = {pace['r2']:.3f}")

    # build predictions for the pace overlay
    lap_nums = np.arange(summary["LapNumber"].min(), summary["LapNumber"].max() + 1)
    predictions = pd.DataFrame({
        "LapNumber": lap_nums,
        "PredictedTime": pace["model"].predict(lap_nums.reshape(-1, 1)),
    })

    # plots
    fastest = laps.pick_fastest()
    telemetry = get_telemetry(fastest)

    plot_speed_trace(telemetry, title=f"{driver} Speed Trace - {event_name}")
    plot_lap_times(summary, title=f"{driver} Lap Times - {event_name}")
    plot_tyre_degradation(tyre_degradation(laps), title=f"{driver} Tyre Deg - {event_name}")
    plot_stint_pace(summary, model_predictions=predictions, title=f"{driver} Race Pace - {event_name}")


def main():
    """Load an F1 session and run the appropriate analysis based on args."""
    args = build_parser().parse_args()

    print(f"Loading {args.year} {args.race} {args.session}...")
    session = load_session(args.year, args.race, args.session)

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
            # analyse_two_drivers(session, args.drivers[0], args.drivers[1])
            pass
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
