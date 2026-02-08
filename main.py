"""CLI entry point for F1 telemetry analysis."""

import argparse
import sys


def build_parser():
    parser = argparse.ArgumentParser(
        description="F1 Telemetry Analysis & Lap Performance Modelling",
    )
    parser.add_argument("--year", type=int, required=True, help="Season year (2018 onwards)")
    parser.add_argument("--race", type=str, required=True, help='Grand Prix name (e.g. "Bahrain") or round number')
    parser.add_argument(
        "--session", type=str, default="R",
        choices=["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"],
        help="Session type (default: R)",
    )
    parser.add_argument(
        "--drivers", nargs="+", metavar="DRIVER",
        help="One or two driver abbreviations (e.g. VER NOR). If omitted, analyses the session winner.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    print(f"Year: {args.year}, Race: {args.race}, Session: {args.session}, Drivers: {args.drivers}")
