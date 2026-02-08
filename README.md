# F1 Telemetry Analysis & Lap Performance Modelling

Telemetry analysis tool for Formula 1 race data. Loads session data via the [FastF1](https://docs.fastf1.dev/) library, runs lap performance analysis, models tyre degradation and lap time behaviour, and generates visualisations comparing driver performance.

Built with Python 3.9+.

## What it does

- Loads F1 session data (practice, qualifying, race) for any season from 2018 onwards
- Breaks down lap times by sector, compound, and stint
- Compares driver telemetry (speed, throttle, brake, gear) on a lap-by-lap basis
- Models tyre degradation curves per compound and stint length
- Fits lap time models to estimate pace trends across a race
- Outputs plots: speed traces, lap time scatter plots, tyre deg curves, driver telemetry comparisons

## Project structure

```
F1-telemetry-analysis/
├── f1_telemetry/
│   ├── __init__.py
│   ├── loader.py          # Session & telemetry data loading
│   ├── analysis.py        # Lap performance & sector analysis
│   ├── modelling.py       # Lap time prediction & tyre deg models
│   └── visualisation.py   # Plotting & chart generation
├── examples/
│   └── race_analysis.py   # Full pipeline example
├── output/                # Generated plots land here
├── main.py                # CLI entry point
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

Requires **Python 3.9** or newer. FastF1 supports Python 3.9 through 3.14.

```bash
# clone the repo
git clone https://github.com/ziyadsaf/F1-telemetry-analysis.git
cd F1-telemetry-analysis

# create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# install dependencies
pip install -r requirements.txt
```

## Usage

Run an analysis from the command line:

```bash
# analyse a specific race
python main.py --year 2024 --race "Bahrain" --session R

# compare two drivers in qualifying
python main.py --year 2024 --race "Monaco" --session Q --drivers VER NOR

# run the example script
python examples/race_analysis.py
```

Or use the modules directly in your own scripts:

```python
from f1_telemetry.loader import load_session, get_driver_laps, get_telemetry
from f1_telemetry.analysis import compare_drivers, analyse_tyre_stints
from f1_telemetry.visualisation import plot_speed_trace, plot_lap_times

session = load_session(2024, "Bahrain", "R")
laps = get_driver_laps(session, "VER")
telemetry = get_telemetry(laps.pick_fastest())

plot_speed_trace(telemetry, title="Verstappen - Bahrain 2024")
```

## Data source

All data comes from the [FastF1](https://docs.fastf1.dev/) library (v3.7+), which pulls from the official F1 live timing API and the Jolpica-F1 API. FastF1 caches data locally after the first load so repeated runs are fast.

No API keys required.

## Dependencies

- **fastf1** >= 3.7.0 — F1 data access and caching
- **pandas** — data manipulation
- **numpy** — numerical operations
- **matplotlib** — plotting
- **seaborn** — statistical visualisation
- **scikit-learn** — lap time modelling and regression
- **tqdm** — progress bars

Full list in `requirements.txt`.

## Output examples

Running the analysis generates plots in the `output/` directory:

- `speed_trace.png` — throttle/brake/speed overlay for a lap
- `lap_times.png` — lap time scatter plot across a race stint
- `tyre_deg.png` — degradation curves by compound
- `driver_comparison.png` — head-to-head telemetry comparison
- `race_pace.png` — race pace scatter with model fit overlay

## License

MIT — see [LICENSE](LICENSE) for details.
