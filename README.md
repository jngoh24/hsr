# ⚡ Relative High-Speed Running — FIFA World Cup 2022

[![Live Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen?style=flat-square&logo=streamlit)](https://hsr-worldcup-2022-gfveg8myjojscddwtja6kc.streamlit.app/)
[![Tests](https://img.shields.io/badge/Tests-13%20passing-brightgreen?style=flat-square&logo=pytest)](tests/test_hsr_metric.py)
[![Data](https://img.shields.io/badge/Data-82M%20frames-blue?style=flat-square)](https://gradientai.io/)
[![Stack](https://img.shields.io/badge/Stack-Databricks%20%7C%20Delta%20Lake%20%7C%20Streamlit-orange?style=flat-square)](notebooks/)

**→ [Open the live dashboard](https://hsr-worldcup-2022-gfveg8myjojscddwtja6kc.streamlit.app/)**

A data science project that challenges the industry standard definition of high-speed running using GradientSports broadcast tracking data from the 2022 FIFA Men's World Cup — processed through an Azure Databricks medallion pipeline and deployed as an interactive Streamlit dashboard.

---

## The problem

> *"The flat 20 km/h threshold treats a midfielder running at 96% of their capacity identically to a centre-back coasting at 73% of theirs. That's not measuring effort — it's measuring speed."*

The conventional HSR definition uses a flat 20 km/h threshold across all players. This was directly inspired by the [GradientSports MLS 2024 HSR analysis](https://www.blog.fc.pff.com/blog/pff-fc-mlsphysical-2024) — published by the same provider whose open tracking data powers this project. Their analysis used the flat threshold. **This project uses the same data to challenge it.**

| Scenario | Speed | Flat threshold | Relative threshold |
|----------|-------|---------------|-------------------|
| Midfielder (v-max 23 km/h) running at 22 km/h | 22 km/h | ✅ counts | ✅ counts (96% of max) |
| Centre-back (v-max 30 km/h) running at 22 km/h | 22 km/h | ✅ counts | ❌ doesn't count (73% of max) |
| Slow player (v-max 18 km/h) running at 15 km/h | 15 km/h | ❌ misses | ✅ counts (83% of max) |

**Key finding:** The industry standard has a one-directional bias at elite level — it never overcounts effort, it only misses it.

---

## The dashboard

**→ [hsr-worldcup-2022.streamlit.app](https://hsr-worldcup-2022-gfveg8myjojscddwtja6kc.streamlit.app/)**

Every chart recalculates dynamically as filters change — no page reloads, no pre-aggregated snapshots.

**The threshold slider is the centrepiece.** Drag it from 60% to 95% and every ranking, heatmap, and comparison table recalculates in real time from the raw run-level data.

| Tab | What it shows |
|-----|--------------|
| **Players** | Top N bar chart · speed vs volume scatter · **30-zone per-player pitch heatmap** · full table |
| **Teams** | HSR volume ranking · speed profile · HSR distance · team summary |
| **Positions** | Bar chart · radar chart across 5 metrics · position table |
| **Definition Comparison** | Who gains/loses runs vs the 20 km/h standard · team outperformance |
| **Pitch Map** | 9-zone heatmap of run start positions across the tournament |
| **Match Analysis** | Per-match top performers · team comparison · pitch heatmap |
| **Tournament Phases** | Group Stage vs Knockout · week-by-week intensity trends |

**Sidebar filters** (apply globally across all tabs):
`Threshold %` · `Team` · `Position` · `Min games` · `Match`

The **per-player zone heatmap** is the most analytically distinctive feature — select any player and see exactly where their high-speed runs start across a 30-zone grid. Directly inspired by the GradientSports MLS article's per-player analysis.

---

## The data

**Source:** [GradientSports FIFA World Cup 2022](https://gradientai.io/) — open broadcast tracking data loaded via [`fast-forward-football`](https://github.com/UnravelSports/fast-forward-football)

```
64 games  ·  32 teams  ·  82,231,917 tracking frames  ·  25 Hz
```

---

## The pipeline

```
┌─────────────────────────────────────────────────────┐
│  GradientSports World Cup 2022 open tracking data   │
└──────────────────────────┬──────────────────────────┘
                           │  fast-forward-football (Polars)
┌──────────────────────────▼──────────────────────────┐
│              Azure Databricks                        │
│                                                      │
│  Bronze  →  Silver  →  Gold                         │
│  82M raw     + speed      + HSR runs                │
│  frames      + v-max      + comparison               │
│              + HSR flag   + player summary           │
└──────────────────────────┬──────────────────────────┘
                           │  CSV export
┌──────────────────────────▼──────────────────────────┐
│         Streamlit Community Cloud                    │
│         Interactive dashboard · public URL           │
└─────────────────────────────────────────────────────┘
```

### Bronze — Raw ingestion

Loads all 64 games into Delta Lake. The key engineering challenge: converting 1.3M+ rows per game from Polars to Spark in-memory caused OOM errors on a single-node 8.8 GB cluster. The fix — Polars writes each game to Parquet on DBFS independently (no JVM involvement), then Spark reads all 64 files in one batch. Eliminated the OOM completely.

### Silver — Feature engineering

Per-frame speed derived from x/y displacement at 25 Hz. v-max estimated at **p99.9** of all speeds pooled across all 64 games.

> **Why p99.9?** Initial choice of p99.5 produced a median v-max of only 20 km/h — an artefact of broadcast tracking where most frames capture walking or jogging. p99.9 (1 in 1,000 frames) gives a median of 24.7 km/h, better reflecting genuine peak capability while remaining robust to noise spikes.

### Gold — Metric tables

HSR frames grouped into discrete run events (≥ 1 second). All tables enriched with player names, team names, and positions. Comparison against the flat 20 km/h standard computed for every player.

---

## Metric definition

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Relative threshold | **80%** of personal v-max | High-intensity but sub-sprint; anchored to individual physiology |
| v-max estimation | **p99.9** across all 64 games | Robust to noise; appropriate for sparse broadcast tracking |
| Minimum run duration | **1 second** (25 frames at 25 Hz) | Filters noise; consistent with industry convention |
| Min frames for v-max | **250** (~10 seconds) | Excludes players with insufficient tracking coverage |
| Speed clip ceiling | **45 km/h** | Guards against frame-gap teleportation artefacts |

**Position rollup:**

| Codes | Group |
|-------|-------|
| GK | GK |
| CB, LCB, RCB, LB, RB, LWB, RWB | DEF |
| CDM, CM, LCM, RCM, CAM, LAM, RAM, LM, RM | MID |
| LW, RW, LF, RF, CF, SS, ST | FWD |

---

## Core metric code

`src/hsr_metric.py` — pure Polars, no Spark dependency, fully testable locally.

```python
compute_speed(df)              # Euclidean x/y displacement → km/h per frame
compute_vmax(df, percentile)   # p99.9 across all games per player
tag_relative_hsr_frames(df)    # flag frames exceeding relative threshold
extract_hsr_runs(df)           # group consecutive frames into run events
summarise_hsr_per_player(df)   # tournament-level aggregation
```

**13 unit tests** — canonical two-player scenarios, missed runs below 20 km/h, duration gating, ball exclusion. All passing.

```
pytest tests/test_hsr_metric.py -v
```

---

## Infrastructure

| Component | Detail |
|-----------|--------|
| Cluster | Azure Databricks · Standard_D4ds_v5 · single-node · 8.8 GB RAM |
| Runtime | Databricks 17.2.x · Spark 3.5 · Python 3.11 |
| Storage | Azure DBFS · Delta Lake (Bronze / Silver / Gold) |
| Compute | PySpark + Polars · `applyInPandas` for distributed UDFs |
| Dashboard | Streamlit Community Cloud |

---

## Repo

```
hsr-worldcup-2022/
├── notebooks/
│   ├── 01_bronze_ingestion.py
│   ├── 02_silver_features.py
│   ├── 03_gold_metrics.py
│   └── 04_gk_investigation.py
├── src/
│   ├── hsr_metric.py
│   └── hsr_comparison.py
├── tests/
│   └── test_hsr_metric.py
├── app/
│   ├── app.py
│   └── data/
│       ├── match_metadata.csv
│       ├── hsr_player_summary.csv
│       ├── hsr_comparison.csv
│       ├── hsr_runs.csv
│       └── hsr_absolute_runs.csv
├── .streamlit/
│   └── config.toml
└── requirements.txt
```

---

*Data: GradientSports FIFA World Cup 2022 open dataset.*  
*Built for the USSF Data Scientist application.*
