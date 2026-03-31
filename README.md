# HSR Metric Dashboard · FIFA World Cup 2022

Interactive dashboard for the **Relative High-Speed Running** metric built on
GradientSports broadcast tracking data from the 2022 FIFA Men's World Cup.

## What this project does

Redefines "high-speed running" using each player's personal maximum speed
rather than the industry-standard flat 20 km/h threshold.

**New definition**: a run where a player reaches ≥ 75% of their personal
recorded v-max and maintains it for at least 1 second.

**Why it matters**: the flat 20 km/h threshold systematically undercounts
high-effort runs from players whose physical ceiling sits below 26.7 km/h —
misrepresenting their contribution in any load or tactical analysis.

## Dashboard features

- **Threshold slider** — adjust the % of v-max threshold and see all charts
  update in real time
- **Player ranking** — top players by HSR runs per game, coloured by intensity
- **Team analysis** — squad-level HSR volume, speed profiles, distance covered
- **Position analysis** — radar chart and bar chart by position group
- **Definition comparison** — who gains/loses runs vs the 20 km/h standard,
  which teams outperform the industry metric
- **Pitch zones** — run distribution by third of the pitch, pitch map of
  run start positions coloured by peak speed

## Data pipeline

```
GradientSports World Cup 2022 open data
    → fast-forward-football (Python, Polars)
        → Databricks Bronze/Silver/Gold Delta tables
            → CSV exports in data/
                → This Streamlit app
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to share.streamlit.io
3. Click "New app" → select this repo → set main file to `app.py`
4. Deploy — you get a public URL instantly

## Data files

Place these three CSV files in the `data/` folder before running:

| File | Source | Rows |
|------|--------|------|
| `hsr_player_summary.csv` | Gold Delta table | ~500 players |
| `hsr_comparison.csv` | Gold Delta table | ~500 players |
| `hsr_runs.csv` | Gold Delta table | ~100k+ run events |

Export from Databricks using the export cell in Notebook 03.

## Tech stack

- **Tracking data**: GradientSports / PFF FIFA World Cup 2022 (open)
- **Data loader**: `fast-forward-football` (Rust-powered, Polars)
- **Pipeline**: Azure Databricks, Delta Lake, PySpark
- **App**: Streamlit, Plotly
- **Hosting**: Streamlit Community Cloud
