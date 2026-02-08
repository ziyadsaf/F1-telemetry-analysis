# F1 Telemetry Analysis

Tool for analyzing Formula 1 race data. Pulls session data from [FastF1](https://docs.fastf1.dev/), breaks down lap times, models tyre wear, and generates plots comparing driver performance.

## What it does

- Load F1 sessions from 2018 onwards
- Compare lap times by sector and stint
- Plot driver telemetry (speed, throttle, brake, gear)
- Model tyre degradation and lap time trends
- Generate visualisations and save them as PNG files

## Project structure

```
F1-telemetry-analysis/
├── f1_telemetry/
│   ├── loader.py          # Session & telemetry data loading
│   ├── analysis.py        # Lap performance & sector analysis
│   ├── modelling.py       # Lap time prediction & tyre deg models
│   └── visualisation.py   # Plotting & chart generation
├── output/                # Generated plots land here
├── main.py                # CLI entry point
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

Requires Python 3.9+.

```bash
git clone https://github.com/ziyadsaf/F1-telemetry-analysis.git
cd F1-telemetry-analysis

python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Usage

```bash
python main.py --year 2024 --race "Bahrain" --session R

python main.py --year 2024 --race "Monaco" --session Q --drivers VER NOR
```

Or import the modules directly:

```python
from f1_telemetry.loader import load_session, get_driver_laps
from f1_telemetry.analysis import compare_drivers
from f1_telemetry.visualisation import plot_speed_trace

session = load_session(2024, "Bahrain", "R")
laps = get_driver_laps(session, "VER")
plot_speed_trace(get_telemetry(laps.pick_fastest()))
```

## Data

Data comes from [FastF1](https://docs.fastf1.dev/) (v3.7+). Gets cached locally after the first load. No API key needed.

## Dependencies

- fastf1 >= 3.7
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tqdm

See `requirements.txt` for full list.

## Output examples

Running the analysis generates plots in the `output/` directory:

- `speed_trace.png` — throttle/brake/speed overlay for a lap
- `lap_times.png` — lap time scatter plot across a race stint
- `tyre_deg.png` — degradation curves by compound
- `driver_comparison.png` — head-to-head telemetry comparison
- `race_pace.png` — race pace scatter with model fit overlay

## License

MIT — see [LICENSE](LICENSE) for details.
