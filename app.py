import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HSR Metric · FIFA World Cup 2022",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    color: #1a1a1a;
}

.stApp {
    background-color: #f7f7f5;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e5e3;
}
section[data-testid="stSidebar"] * {
    color: #1a1a1a !important;
}

/* ── Headers ── */
h1 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-weight: 600;
    font-size: 28px;
    color: #111111;
    letter-spacing: -0.01em;
    line-height: 1.2;
}
h2, h3, h4 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #111111;
    letter-spacing: -0.01em;
}
h3 { font-size: 16px; }
h4 { font-size: 14px; font-weight: 500; color: #444; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #e5e5e3;
    border-radius: 4px;
    padding: 16px 20px;
    box-shadow: none;
}
[data-testid="metric-container"] label {
    font-size: 11px;
    font-weight: 500;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'Inter', sans-serif;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 26px;
    font-weight: 600;
    color: #111 !important;
    font-family: 'Source Serif 4', serif;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 2px solid #e5e5e3;
    gap: 0;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #888;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background-color: transparent !important;
    color: #111 !important;
    border-bottom: 2px solid #111 !important;
}

/* ── Divider ── */
hr { border-color: #e5e5e3; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e5e5e3;
    border-radius: 4px;
    background: #fff;
}

/* ── Slider label ── */
[data-testid="stSlider"] label {
    font-size: 12px;
    font-weight: 500;
    color: #444 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] label {
    font-size: 12px;
    font-weight: 500;
    color: #444 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Caption / label utility classes ── */
.eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.kicker {
    font-family: 'Source Serif 4', serif;
    font-size: 13px;
    font-style: italic;
    color: #555;
}
.threshold-badge {
    display: inline-block;
    background: #111;
    color: #fff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    padding: 3px 10px;
    border-radius: 3px;
    letter-spacing: 0.02em;
}
.gained-badge { display:inline-block; background:#e8f5e9; color:#2e7d32;
                font-size:11px; font-weight:600; padding:2px 8px;
                border-radius:3px; font-family:'Inter',sans-serif; }
.lost-badge   { display:inline-block; background:#fce4e4; color:#c62828;
                font-size:11px; font-weight:600; padding:2px 8px;
                border-radius:3px; font-family:'Inter',sans-serif; }
.unch-badge   { display:inline-block; background:#f5f5f5; color:#666;
                font-size:11px; font-weight:600; padding:2px 8px;
                border-radius:3px; font-family:'Inter',sans-serif; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_data():
    summary    = pd.read_csv(os.path.join(DATA_DIR, "hsr_player_summary.csv"))
    comparison = pd.read_csv(os.path.join(DATA_DIR, "hsr_comparison.csv"))
    runs       = pd.read_csv(os.path.join(DATA_DIR, "hsr_runs.csv"))
    meta_path  = os.path.join(DATA_DIR, "match_metadata.csv")
    metadata   = pd.read_csv(meta_path) if os.path.exists(meta_path) else pd.DataFrame()
    return summary, comparison, runs, metadata

try:
    summary_df, comparison_df, runs_df, match_meta_df = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    match_meta_df = pd.DataFrame()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p class="eyebrow" style="margin-bottom:2px;">HSR Metric</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-family:Inter;font-size:12px;color:#666;margin:0 0 16px 0;">'
        'FIFA World Cup 2022</p>',
        unsafe_allow_html=True
    )
    st.divider()
    st.markdown(
        '<p class="eyebrow" style="margin-bottom:8px;">Threshold</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-family:Inter;font-size:12px;color:#666;margin:0 0 8px 0;">'
        'Minimum % of personal v-max to count as a high-speed run</p>',
        unsafe_allow_html=True
    )

    threshold_int = st.slider(
        label="Relative threshold",
        min_value=60,
        max_value=95,
        value=75,
        step=5,
        format="%d%%",
        help="75% = industry-comparable. Raise to see only near-maximum efforts."
    )
    threshold_pct = threshold_int / 100.0

    st.markdown(f"""
    <div style="background:#f0f0ee;border-left:3px solid #111;padding:10px 14px;margin-top:8px;">
        <p style="font-family:Inter;font-size:11px;font-weight:600;color:#888;
                  text-transform:uppercase;letter-spacing:0.06em;margin:0 0 2px 0;">Active threshold</p>
        <p style="font-family:'Source Serif 4',serif;font-size:28px;font-weight:600;
                  color:#111;margin:0;">{threshold_pct*100:.0f}%</p>
        <p style="font-family:Inter;font-size:11px;color:#666;margin:2px 0 0 0;">of personal v-max</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Filters")

    if data_loaded:
        teams = sorted(summary_df["team_short"].dropna().unique().tolist())
        selected_teams = st.multiselect(
            "Teams",
            options=teams,
            default=teams,
            placeholder="All teams"
        )

        pos_groups = ["GK", "DEF", "MID", "FWD", "UNK"]
        selected_positions = st.multiselect(
            "Position group",
            options=pos_groups,
            default=["GK", "DEF", "MID", "FWD"],
            placeholder="All positions"
        )

        min_games = st.slider("Min games played", 1, 7, 2)

    st.divider()
    st.markdown(
        '<p class="eyebrow" style="margin-bottom:8px;">Match filter</p>',
        unsafe_allow_html=True
    )
    if data_loaded and not match_meta_df.empty:
        match_meta_df["label"] = (
            match_meta_df["home_team_short"] + " vs " +
            match_meta_df["away_team_short"] + "  ·  W" +
            match_meta_df["week"].astype(str) +
            "  (" + match_meta_df["Round"].str[:2].str.upper() + ")"
        )
        game_options = ["All games"] + match_meta_df.sort_values("date")["label"].tolist()
        selected_game_label = st.selectbox(
            "Game",
            options=game_options,
            help="Filter all tabs to a single match"
        )
        if selected_game_label != "All games":
            selected_game_id = str(match_meta_df[
                match_meta_df["label"] == selected_game_label
            ]["game_id"].iloc[0])
        else:
            selected_game_id = None
    else:
        selected_game_id = None

    st.divider()
    st.markdown(
        '<p style="font-family:Inter;font-size:11px;color:#888;line-height:1.5;">'
        '<strong>New definition</strong> — a run where a player reaches &ge; threshold% '
        'of their personal v-max for &ge; 1 second.<br><br>'
        '<strong>Industry standard</strong> — flat 20 km/h absolute threshold.</p>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(
    '<p class="eyebrow" style="margin-bottom:4px;">FIFA Men\'s World Cup 2022 · GradientSports Tracking</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<h1 style="margin:0 0 4px 0;">Relative High-Speed Running</h1>',
    unsafe_allow_html=True
)
st.markdown(
    f"<p class='kicker'>Redefining high-speed running as &ge; <span class='threshold-badge'>{threshold_pct*100:.0f}%</span> of each player's personal v-max, sustained for &ge; 1 second. Industry standard shown for comparison.</p>",
    unsafe_allow_html=True
)

st.divider()

if not data_loaded:
    st.error(
        "⚠️ Data files not found. Place `hsr_player_summary.csv`, "
        "`hsr_comparison.csv`, and `hsr_runs.csv` in the `data/` folder."
    )
    st.stop()

# ─────────────────────────────────────────────
# Position rollup
# ─────────────────────────────────────────────
POS_ROLLUP = {
    # Goalkeeper
    "GK":  "GK",
    # Defenders
    "CB":  "DEF", "LCB": "DEF", "RCB": "DEF",
    "LB":  "DEF", "RB":  "DEF",
    "LWB": "DEF", "RWB": "DEF", "SW":  "DEF",
    # Midfielders
    "CDM": "MID", "CM":  "MID", "LCM": "MID", "RCM": "MID",
    "CAM": "MID", "LAM": "MID", "RAM": "MID",
    "LM":  "MID", "RM":  "MID",
    # Forwards
    "LW":  "FWD", "RW":  "FWD",
    "LF":  "FWD", "RF":  "FWD",
    "CF":  "FWD", "SS":  "FWD", "ST": "FWD",
    # Unknown
    "UNK": "UNK",
}

def add_pos_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add pos (rolled-up group) and pos_detail (original code) columns."""
    df = df.copy()
    df["pos_detail"] = df["position"]
    df["pos"] = df["position"].map(POS_ROLLUP).fillna("UNK")
    return df

summary_df   = add_pos_columns(summary_df)
comparison_df= add_pos_columns(comparison_df)
# Always add pos columns to runs_df — position col may be named differently
if "position" in runs_df.columns:
    runs_df = add_pos_columns(runs_df)
elif "pos_detail" not in runs_df.columns:
    runs_df["pos_detail"] = "UNK"
    runs_df["pos"] = "UNK"

# ─────────────────────────────────────────────
# Recompute metric at selected threshold
# ─────────────────────────────────────────────
# Recompute runs_per_game dynamically from runs_df using pct_of_vmax
# runs_df has one row per run with pct_of_vmax — filter to runs that
# qualify at the current threshold and recount per player per game

qualifying_runs = runs_df[runs_df["pct_of_vmax"] >= threshold_pct].copy()
if selected_game_id:
    qualifying_runs = qualifying_runs[qualifying_runs["game_id"].astype(str) == selected_game_id]

# Ensure qualifying_runs has team_short and pos for pitch map filtering
# Merge from player_lookup if columns are missing
_player_meta = summary_df[["player_id", "team_short", "pos", "pos_detail",
                             "player_name"]].drop_duplicates("player_id")     if "player_id" in summary_df.columns else None

if _player_meta is not None and "player_id" in qualifying_runs.columns:
    _missing = [c for c in ["team_short", "pos", "player_name"]
                if c not in qualifying_runs.columns]
    if _missing:
        qualifying_runs = qualifying_runs.merge(
            _player_meta[["player_id"] + _missing],
            on="player_id", how="left"
        )

dynamic_counts = (
    qualifying_runs
    .groupby("player_id")
    .agg(
        total_runs_dynamic   = ("run_id", "count"),
        games_with_runs      = ("game_id", "nunique"),
        mean_pct_dynamic     = ("pct_of_vmax", "mean"),
        mean_duration_dynamic= ("duration_sec", "mean"),
        total_distance_dynamic=("distance_m", "sum"),
        mean_peak_dynamic    = ("peak_speed_kmh", "mean"),
    )
    .reset_index()
)
dynamic_counts["runs_per_game_dynamic"] = (
    dynamic_counts["total_runs_dynamic"]
    / dynamic_counts["games_with_runs"].clip(lower=1)
).round(2)
dynamic_counts["mean_pct_dynamic"] = (
    dynamic_counts["mean_pct_dynamic"] * 100
).round(1)

# Merge dynamic counts back onto summary
summary_merged = summary_df.merge(dynamic_counts, on="player_id", how="left")
summary_merged["total_runs_dynamic"]    = summary_merged["total_runs_dynamic"].fillna(0).astype(int)
summary_merged["runs_per_game_dynamic"] = summary_merged["runs_per_game_dynamic"].fillna(0)
summary_merged["mean_peak_dynamic"]     = summary_merged["mean_peak_dynamic"].fillna(0)
summary_merged["threshold_at_pct"]     = summary_merged["vmax_kmh"] * threshold_pct

# Recompute comparison metrics at new threshold
comparison_df["threshold_at_pct"]   = comparison_df["vmax_kmh"] * threshold_pct
comparison_df["above_absolute"]     = comparison_df["threshold_at_pct"] <= 20.0
comparison_df["new_threshold_vs_20"]= comparison_df["threshold_at_pct"] - 20.0

# Dynamic comparison: recount absolute and relative runs at new threshold
abs_counts = (
    runs_df[runs_df["peak_speed_kmh"] >= 20.0]
    .groupby("player_id")
    .size()
    .reset_index(name="runs_absolute_dynamic")
)
rel_counts = (
    qualifying_runs
    .groupby("player_id")
    .size()
    .reset_index(name="runs_relative_dynamic")
)
comparison_df = comparison_df.merge(abs_counts, on="player_id", how="left")
comparison_df = comparison_df.merge(rel_counts, on="player_id", how="left")
comparison_df["runs_absolute_dynamic"] = comparison_df["runs_absolute_dynamic"].fillna(0).astype(int)
comparison_df["runs_relative_dynamic"] = comparison_df["runs_relative_dynamic"].fillna(0).astype(int)
comparison_df["run_delta"]   = comparison_df["runs_relative_dynamic"] - comparison_df["runs_absolute_dynamic"]
comparison_df["pct_change"]  = (
    (comparison_df["runs_relative_dynamic"] - comparison_df["runs_absolute_dynamic"])
    / comparison_df["runs_absolute_dynamic"].clip(lower=1) * 100
).round(1)
comparison_df["category"] = comparison_df["run_delta"].apply(
    lambda x: "gained" if x > 0 else ("lost" if x < 0 else "unchanged")
)

# Apply sidebar filters
filtered_summary = summary_merged[
    summary_merged["team_short"].isin(selected_teams) &
    summary_merged["pos"].isin(selected_positions) &
    (summary_merged["games_appeared"] >= min_games) &
    (~summary_merged["low_confidence"])
].copy()

filtered_comparison = comparison_df[
    comparison_df["team_short"].isin(selected_teams) &
    comparison_df["pos"].isin(selected_positions)
].copy()

# ─────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

n_below_20 = (comparison_df["threshold_at_pct"] < 20.0).sum()
pct_below  = n_below_20 / len(comparison_df) * 100
avg_vmax   = filtered_summary["vmax_kmh"].mean()
avg_runs   = filtered_summary["runs_per_game_dynamic"].mean()
top_speed  = filtered_summary["tournament_peak_speed_kmh"].max()
n_players  = len(filtered_summary)

k1.metric("Players tracked",       f"{n_players:,}")
k2.metric("Avg v-max",             f"{avg_vmax:.1f} km/h")
k3.metric("Avg HSR runs / game",   f"{avg_runs:.1f}")
k4.metric("Top speed recorded",    f"{top_speed:.1f} km/h")
k5.metric("Players below 20 km/h threshold", f"{n_below_20} ({pct_below:.0f}%)",
          help="Players whose relative threshold sits below 20 km/h — "
               "meaning the industry standard misses their high-effort runs")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Plotly theme
# ─────────────────────────────────────────────
PLOT_BG    = "#ffffff"
PAPER_BG   = "#f7f7f5"
GRID_COLOR = "#eeeeec"
TEXT_COLOR = "#666666"
ACCENT     = "#1a6b3c"   # Athletic-style dark green
BLUE       = "#1a4b8c"   # deep blue for second accent
RED        = "#c0392b"   # muted red
AMBER      = "#b7791f"   # warm amber
GREEN      = "#1a6b3c"
POS_COLORS = {
    "GK":  "#f0a500",  # amber
    "DEF": "#1a4b8c",  # blue
    "MID": "#1a6b3c",  # green
    "FWD": "#c0392b",  # red
    "UNK": "#aaaaaa",
}

def base_layout(title="", height=400, xaxis=None, yaxis=None):
    default_axis = dict(gridcolor=GRID_COLOR, showline=False, zeroline=False)
    x = {**default_axis, **(xaxis or {})}
    y = {**default_axis, **(yaxis or {})}
    return dict(
        title=dict(text=title, font=dict(family="Inter", size=13, color="#111111")),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="DM Mono", color=TEXT_COLOR, size=11),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=x,
        yaxis=y,
    )

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Players",
    "Teams",
    "Positions",
    "Definition Comparison",
    "Pitch Map",
    "Match Analysis",
    "Tournament Phases",
])

# ══════════════════════════════════════════════
# TAB 1 — Player ranking
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### Top players by HSR runs per game")
    st.markdown(
        f'<p class="caption">Relative definition · threshold = {threshold_pct*100:.0f}% of personal v-max · min {min_games} games</p>',
        unsafe_allow_html=True
    )

    col_chart, col_scatter = st.columns([3, 2])

    with col_chart:
        top_n = st.select_slider("Show top players", options=[10, 15, 20, 25, 30], value=20,
                                  format_func=lambda x: f"Top {x}")
        top_players = filtered_summary.nlargest(top_n, "runs_per_game_dynamic")

        fig_bar = go.Figure()
        pos_color_map = top_players["pos"].map(POS_COLORS).fillna("#aaa")
        fig_bar.add_trace(go.Bar(
            x=top_players["runs_per_game_dynamic"],
            y=top_players["player_name"] + " (" + top_players["team_short"] + ")",
            orientation="h",
            marker=dict(
                color=pos_color_map,
                line=dict(width=0),
                opacity=0.85,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Runs/game: %{x:.1f}<br>"
                "<extra></extra>"
            ),
        ))
        fig_bar.update_layout(
            **base_layout(
                height=max(400, top_n * 22),
                yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                xaxis=dict(title="HSR runs per game"),
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_scatter:
        st.markdown("#### Speed profile vs HSR volume")
        fig_scatter = px.scatter(
            filtered_summary,
            x="vmax_kmh",
            y="runs_per_game_dynamic",
            color="pos",
            color_discrete_map=POS_COLORS,
            size="games_appeared",
            hover_data=["player_name", "team_name", "pos_detail", "tournament_peak_speed_kmh"],
            labels={
                "vmax_kmh": "Personal v-max (km/h)",
                "runs_per_game_dynamic": "HSR runs per game",
                "pos": "Position group",
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_scatter.update_layout(**base_layout(height=420))
        fig_scatter.update_traces(marker=dict(line=dict(width=0)))
        # Add threshold line
        fig_scatter.add_vline(
            x=20 / threshold_pct,
            line=dict(color=AMBER, width=1, dash="dot"),
            annotation_text=f"v-max where {threshold_pct*100:.0f}% = 20 km/h",
            annotation_font=dict(size=9, color=AMBER),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)



    # ══ Per-player zone heatmap ══════════════════════════════════════════
    st.markdown("#### Player zone heatmap")
    st.markdown(
        '<p style="font-family:Inter;font-size:12px;color:#666;">'
        'Select a player to see where their high-speed runs start across 30 pitch zones '
        '(6 columns x 5 rows). Darker zones = more runs.</p>',
        unsafe_allow_html=True
    )

    if "start_x" in runs_df.columns and "player_name" in qualifying_runs.columns:

        player_options = (
            filtered_summary
            .sort_values("runs_per_game_dynamic", ascending=False)
            .dropna(subset=["player_name"])
            ["player_name"].tolist()
        )

        ph_col1, ph_col2 = st.columns([1, 2])

        with ph_col1:
            selected_player = st.selectbox(
                "Select player",
                options=player_options,
                key="player_heatmap_selector"
            )

            pmeta = filtered_summary[filtered_summary["player_name"] == selected_player]
            if not pmeta.empty:
                pm = pmeta.iloc[0]
                st.markdown(
                    f'<div style="background:#f0f0ee;border-left:3px solid #111;'
                    f'padding:10px 14px;margin-top:8px;">'
                    f'<p style="font-family:Inter;font-size:11px;font-weight:600;color:#888;'
                    f'text-transform:uppercase;letter-spacing:0.06em;margin:0 0 4px 0;">'
                    f'{pm.get("team_name","")}</p>'
                    f'<p style="font-family:Georgia,serif;font-size:20px;font-weight:600;'
                    f'color:#111;margin:0 0 8px 0;">{selected_player}</p>'
                    f'<p style="font-family:Inter;font-size:12px;color:#444;margin:0;">'
                    f'{pm.get("pos_detail","")} &nbsp;&middot;&nbsp; '
                    f'v-max <strong>{pm.get("vmax_kmh",0):.1f} km/h</strong></p>'
                    f'<p style="font-family:Inter;font-size:12px;color:#444;margin:4px 0 0 0;">'
                    f'<strong>{int(pm.get("total_runs_dynamic",0))}</strong> HSR runs &nbsp;&middot;&nbsp; '
                    f'<strong>{pm.get("runs_per_game_dynamic",0):.1f}</strong> per game</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        with ph_col2:
            player_runs = qualifying_runs[
                qualifying_runs["player_name"] == selected_player
            ].copy()

            if player_runs.empty:
                st.info("No qualifying runs found for this player at the current threshold.")
            else:
                # 30-zone grid: 6 cols x 5 rows
                x_edges_30 = [-52.5, -35.0, -17.5,  0.0, 17.5, 35.0, 52.5]
                y_edges_30 = [-34.0, -20.4,  -6.8,  6.8, 20.4, 34.0]

                # Channel labels — one word each to avoid multi-line string issues
                ch_labels_30 = ["Right", "R. Half", "Centre", "L. Half", "Left"]

                zone_counts_30 = np.zeros((5, 6), dtype=int)
                zone_speed_30  = np.zeros((5, 6))

                for _, row in player_runs.iterrows():
                    xi = min(int(np.searchsorted(x_edges_30[1:], row["start_x"])), 5)
                    yi = min(int(np.searchsorted(y_edges_30[1:], row["start_y"])), 4)
                    zone_counts_30[yi, xi] += 1
                    if "peak_speed_kmh" in player_runs.columns:
                        zone_speed_30[yi, xi] += row["peak_speed_kmh"]

                total_pr   = zone_counts_30.sum()
                zone_pct_30 = np.where(total_pr > 0, zone_counts_30 / total_pr * 100, 0)
                max_c = zone_counts_30.max() if zone_counts_30.max() > 0 else 1

                PITCH_BG_P = "#f8f8f5"
                PITCH_LN_P = "#c8c8c4"
                PL_TOP     = "#aaaaaa"

                fig_ph = go.Figure()

                # Pitch background
                fig_ph.add_shape(
                    type="rect", x0=-52.5, x1=52.5, y0=-34, y1=34,
                    fillcolor=PITCH_BG_P,
                    line=dict(color=PITCH_LN_P, width=1.5),
                    layer="below"
                )

                # Zone rectangles
                for row_i in range(5):
                    for col_i in range(6):
                        x0 = x_edges_30[col_i]
                        x1 = x_edges_30[col_i + 1]
                        y0 = y_edges_30[row_i]
                        y1 = y_edges_30[row_i + 1]
                        count = int(zone_counts_30[row_i, col_i])
                        pct   = float(zone_pct_30[row_i, col_i])
                        inten = count / max_c

                        r_c = int(255 + (26  - 255) * inten)
                        g_c = int(255 + (75  - 255) * inten)
                        b_c = int(255 + (140 - 255) * inten)
                        fill = "rgba(%d,%d,%d,0.80)" % (r_c, g_c, b_c)
                        tcol = "#ffffff" if inten > 0.45 else "#222222"
                        cx   = (x0 + x1) / 2
                        cy   = (y0 + y1) / 2

                        fig_ph.add_shape(
                            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                            fillcolor=fill,
                            line=dict(color=PITCH_LN_P, width=0.5),
                            layer="below"
                        )

                        if count > 0:
                            fig_ph.add_annotation(
                                x=cx, y=cy + 2.5,
                                text="<b>%d</b>" % count,
                                showarrow=False,
                                font=dict(family="Inter", size=12, color=tcol),
                                xanchor="center", yanchor="middle",
                            )
                            fig_ph.add_annotation(
                                x=cx, y=cy - 2.8,
                                text="%.0f%%" % pct,
                                showarrow=False,
                                font=dict(family="Inter", size=9, color=tcol),
                                xanchor="center", yanchor="middle",
                            )

                # Pitch markings
                for sh in [
                    dict(type="line", x0=0, x1=0, y0=-34, y1=34,
                         line=dict(color=PL_TOP, width=1.2)),
                    dict(type="circle", x0=-9.15, x1=9.15, y0=-9.15, y1=9.15,
                         line=dict(color=PL_TOP, width=1)),
                    dict(type="rect", x0=-52.5, x1=-36, y0=-20.16, y1=20.16,
                         line=dict(color=PL_TOP, width=1)),
                    dict(type="rect", x0=36, x1=52.5, y0=-20.16, y1=20.16,
                         line=dict(color=PL_TOP, width=1)),
                    dict(type="rect", x0=-52.5, x1=52.5, y0=-34, y1=34,
                         line=dict(color=PL_TOP, width=1.5)),
                ]:
                    fig_ph.add_shape(**sh)

                # Zone grid lines
                for xb in x_edges_30[1:-1]:
                    fig_ph.add_shape(
                        type="line", x0=xb, x1=xb, y0=-34, y1=34,
                        line=dict(color=PL_TOP, width=0.6, dash="dot")
                    )
                for yb in y_edges_30[1:-1]:
                    fig_ph.add_shape(
                        type="line", x0=-52.5, x1=52.5, y0=yb, y1=yb,
                        line=dict(color=PL_TOP, width=0.6, dash="dot")
                    )

                # Third labels above
                third_labels = ["Defensive third", "Middle third", "Attacking third"]
                third_cx     = [-35.0, 0.0, 35.0]
                for lbl, cx in zip(third_labels, third_cx):
                    fig_ph.add_annotation(
                        x=cx, y=36.2, text=lbl.upper(),
                        showarrow=False,
                        font=dict(family="Inter", size=9, color="#888"),
                        xanchor="center"
                    )

                # Channel labels left
                for i, lbl in enumerate(ch_labels_30):
                    cy = (y_edges_30[i] + y_edges_30[i + 1]) / 2
                    fig_ph.add_annotation(
                        x=-55, y=cy, text=lbl,
                        showarrow=False,
                        font=dict(family="Inter", size=9, color="#888"),
                        xanchor="right"
                    )

                # Attack direction arrow
                fig_ph.add_annotation(
                    x=50, y=-37, ax=38, ay=-37,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1,
                    arrowwidth=1.5, arrowcolor="#888",
                    text="Attack",
                    font=dict(family="Inter", size=9, color="#888"),
                    xanchor="right",
                )

                fig_ph.update_layout(
                    plot_bgcolor=PITCH_BG_P,
                    paper_bgcolor=PAPER_BG,
                    height=460,
                    margin=dict(l=70, r=16, t=30, b=20),
                    xaxis=dict(range=[-58, 55], showgrid=False, zeroline=False,
                               showticklabels=False, showline=False),
                    yaxis=dict(range=[-40, 39], showgrid=False, zeroline=False,
                               showticklabels=False, showline=False,
                               scaleanchor="x", scaleratio=1),
                    showlegend=False,
                )

                st.plotly_chart(fig_ph, use_container_width=True)

                # Top zone narrative
                ch_name   = {0: "right channel", 1: "right half-space",
                             2: "central channel", 3: "left half-space",
                             4: "left channel"}
                third_name = {0: "defensive third", 1: "defensive third",
                              2: "middle third",    3: "middle third",
                              4: "attacking third", 5: "attacking third"}

                flat_idx  = zone_counts_30.flatten().argsort()[::-1]
                top_zones = [(int(idx // 6), int(idx % 6)) for idx in flat_idx
                             if zone_counts_30[idx // 6, idx % 6] > 0][:3]

                if top_zones:
                    parts = []
                    for ri, ci in top_zones:
                        cnt = int(zone_counts_30[ri, ci])
                        pct = float(zone_pct_30[ri, ci])
                        parts.append(
                            "<strong>%d runs (%.0f%%)</strong> in the %s of the %s"
                            % (cnt, pct, ch_name.get(ri, ""), third_name.get(ci, ""))
                        )
                    st.markdown(
                        '<p style="font-family:Inter;font-size:12px;color:#444;'
                        'line-height:1.8;margin-top:4px;">'
                        + " &nbsp;&middot;&nbsp; ".join(parts) + "</p>",
                        unsafe_allow_html=True
                    )
    else:
        st.info("Player zone heatmap requires start_x/start_y columns in hsr_runs.csv.")


    st.divider()
    st.markdown("#### Full player table")

    # Build display table — add avg speed, cap pct_of_vmax at 100%
    display_cols = [
        "player_name", "team_name", "pos", "pos_detail", "games_appeared",
        "vmax_kmh", "threshold_at_pct", "total_runs_dynamic", "runs_per_game_dynamic",
        "hsr_distance_per_game_m", "mean_peak_dynamic", "mean_pct_of_vmax_pct",
        "tournament_peak_speed_kmh",
    ]
    table_df = filtered_summary[display_cols].copy()

    # Cap avg % of v-max at 100 — values above 100 occur when peak speed
    # in a run slightly exceeds the p99.5 v-max estimate (expected, not an error)
    table_df["mean_pct_of_vmax_pct"] = table_df["mean_pct_of_vmax_pct"].clip(upper=100.0)

    # Round numeric columns
    table_df["vmax_kmh"]           = table_df["vmax_kmh"].round(1)
    table_df["threshold_at_pct"]   = table_df["threshold_at_pct"].round(1)
    table_df["hsr_distance_per_game_m"] = table_df["hsr_distance_per_game_m"].round(0)
    table_df["mean_peak_dynamic"]  = table_df["mean_peak_dynamic"].round(1)
    table_df["tournament_peak_speed_kmh"] = table_df["tournament_peak_speed_kmh"].round(1)

    st.dataframe(
        table_df
        .sort_values("runs_per_game_dynamic", ascending=False)
        .reset_index(drop=True)
        .rename(columns={
            "player_name":              "Player",
            "team_name":                "Team",
            "pos":                      "Pos",
            "pos_detail":               "Pos detail",
            "games_appeared":           "Games",
            "vmax_kmh":                 "v-max (km/h)",
            "threshold_at_pct":         "Threshold (km/h)",
            "total_runs_dynamic":       "Total runs",
            "runs_per_game_dynamic":    "Runs / game",
            "hsr_distance_per_game_m":  "HSR dist / game (m)",
            "mean_peak_dynamic":        "Avg speed in run (km/h)",
            "mean_pct_of_vmax_pct":     "Avg % of v-max",
            "tournament_peak_speed_kmh":"Peak speed (km/h)",
        }),
        use_container_width=True,
        height=400,
    )

# ══════════════════════════════════════════════
# TAB 2 — Team analysis
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### Team HSR profile")

    team_agg = (
        filtered_summary
        .groupby(["team_name", "team_short"])
        .agg(
            avg_runs_per_game       = ("runs_per_game_dynamic", "mean"),
            total_team_runs         = ("total_runs_dynamic", "sum"),
            avg_vmax                = ("vmax_kmh", "mean"),
            avg_peak_speed          = ("tournament_peak_speed_kmh", "mean"),
            avg_hsr_distance        = ("hsr_distance_per_game_m", "mean"),
            avg_intensity           = ("mean_pct_of_vmax_pct", "mean"),
            n_players               = ("player_name", "count"),
        )
        .reset_index()
        .sort_values("avg_runs_per_game", ascending=False)
    )

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        fig_team_bar = go.Figure(go.Bar(
            x=team_agg["avg_runs_per_game"],
            y=team_agg["team_short"],
            orientation="h",
            marker=dict(
                color=team_agg["avg_vmax"],
                colorscale=[[0, "#0d2137"], [1, "#4fc3f7"]],
                colorbar=dict(title=dict(text="Avg v-max"), thickness=10),
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>Avg runs/game/player: %{x:.2f}<extra></extra>",
        ))
        fig_team_bar.update_layout(
            **base_layout(
                "Avg HSR runs per game per player",
                height=700,
                yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                xaxis=dict(title="Avg runs per game"),
            )
        )
        st.plotly_chart(fig_team_bar, use_container_width=True)

    with col_t2:
        fig_team_scatter = px.scatter(
            team_agg,
            x="avg_vmax",
            y="avg_runs_per_game",
            text="team_short",
            size="n_players",
            color="avg_intensity",
            color_continuous_scale=[[0, "#ddeeff"], [1, "#1a4b8c"]],
            labels={
                "avg_vmax": "Avg v-max (km/h)",
                "avg_runs_per_game": "Avg HSR runs per game",
                "avg_intensity": "Avg intensity (% v-max)",
            },
        )
        fig_team_scatter.update_traces(
            textposition="top center",
            textfont=dict(size=9, color="#94a3b8"),
            marker=dict(line=dict(width=0)),
        )
        fig_team_scatter.update_layout(**base_layout("Speed vs volume — team view", height=420))
        st.plotly_chart(fig_team_scatter, use_container_width=True)

        # HSR distance per game
        team_agg_sorted = team_agg.sort_values("avg_hsr_distance", ascending=True).tail(16)
        fig_dist = go.Figure(go.Bar(
            x=team_agg_sorted["avg_hsr_distance"],
            y=team_agg_sorted["team_short"],
            orientation="h",
            marker=dict(color=ACCENT, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Avg HSR dist/game: %{x:.0f}m<extra></extra>",
        ))
        fig_dist.update_layout(
            **base_layout(
                "Avg HSR distance per game (m)",
                height=280,
                yaxis=dict(tickfont=dict(size=11)),
                xaxis=dict(title="metres"),
            )
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()
    st.markdown("#### Team summary table")
    st.dataframe(
        team_agg
        .sort_values("avg_runs_per_game", ascending=False)
        .reset_index(drop=True)
        .rename(columns={
            "team_name":                    "Team",
            "team_short":                   "Code",
            "avg_runs_per_game":            "Avg runs / game",
            "total_team_runs":              "Total runs",
            "avg_vmax":                     "Avg v-max (km/h)",
            "avg_peak_speed":               "Avg peak speed (km/h)",
            "avg_hsr_distance":             "Avg HSR dist / game (m)",
            "avg_intensity":                "Avg intensity (% v-max)",
            "n_players_tracked":            "Players tracked",
        })
        .round(2),
        use_container_width=True,
        height=400,
    )

# ══════════════════════════════════════════════
# TAB 3 — Position analysis
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### HSR by position")

    pos_agg = (
        filtered_summary
        .groupby("pos")
        .agg(
            avg_runs_per_game   = ("runs_per_game_dynamic", "mean"),
            avg_vmax            = ("vmax_kmh", "mean"),
            avg_threshold       = ("threshold_kmh", "mean"),
            avg_distance        = ("hsr_distance_per_game_m", "mean"),
            avg_intensity       = ("mean_pct_of_vmax_pct", "mean"),
            avg_peak_speed      = ("tournament_peak_speed_kmh", "mean"),
            n_players           = ("player_name", "count"),
        )
        .reset_index()
        .sort_values("avg_runs_per_game", ascending=False)
        .reset_index()
    )
    pos_agg = pos_agg.rename(columns={"pos": "position"})

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        fig_pos = px.bar(
            pos_agg,
            x="position",
            y="avg_runs_per_game",
            color="avg_intensity",
            color_continuous_scale=[[0, "#ddeeff"], [0.5, "#5588cc"], [1, "#1a4b8c"]],
            text=pos_agg["avg_runs_per_game"].round(1),
            labels={
                "pos": "Position group",
                "avg_runs_per_game": "Avg HSR runs per game",
                "avg_intensity": "Avg intensity",
            },
        )
        fig_pos.update_traces(
            textposition="outside",
            textfont=dict(size=10),
            marker=dict(line=dict(width=0)),
        )
        fig_pos.update_layout(**base_layout("Avg HSR runs per game by position", height=400))
        st.plotly_chart(fig_pos, use_container_width=True)

    with col_p2:
        # Radar chart — position profiles
        categories = ["avg_runs_per_game", "avg_vmax", "avg_distance",
                      "avg_intensity", "avg_peak_speed"]
        labels     = ["Runs/game", "v-max", "HSR dist", "Intensity", "Peak speed"]

        fig_radar = go.Figure()
        colors    = [POS_COLORS.get(p, "#888888") for p in ["GK","DEF","MID","FWD","UNK"]]

        for i, row in pos_agg.iterrows():
            vals = []
            for col in categories:
                col_min = pos_agg[col].min()
                col_max = pos_agg[col].max()
                norm = (row[col] - col_min) / (col_max - col_min + 1e-9)
                vals.append(round(norm * 100, 1))
            vals.append(vals[0])

            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=labels + [labels[0]],
                name=row.get("position", row.name),
                line=dict(color=colors[i % len(colors)], width=2),
                fill="toself",
                fillcolor=colors[i % len(colors)],
                opacity=0.15,
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor="#ffffff",
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor=GRID_COLOR, tickfont=dict(size=8)),
                angularaxis=dict(gridcolor=GRID_COLOR),
            ),
            showlegend=True,
            legend=dict(font=dict(size=10)),
            **base_layout("Position profile (normalised)", height=400),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Position table
    st.markdown("#### Position summary table")
    st.dataframe(
        pos_agg.rename(columns={
            "position": "Pos group",
            "avg_runs_per_game": "Avg runs/game",
            "avg_vmax": "Avg v-max",
            "avg_threshold": "Avg threshold (km/h)",
            "avg_distance": "Avg HSR dist/game (m)",
            "avg_intensity": "Avg intensity (% v-max)",
            "avg_peak_speed": "Avg peak speed",
            "n_players": "N players",
        }).round(2),
        use_container_width=True,
    )

# ══════════════════════════════════════════════
# TAB 4 — Definition comparison
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### Relative vs industry standard (20 km/h flat)")
    st.markdown(
        f'<p class="caption">Comparing {threshold_pct*100:.0f}% of personal v-max '
        f'vs flat 20 km/h threshold across all players</p>',
        unsafe_allow_html=True
    )

    # Summary stats
    fc1, fc2, fc3 = st.columns(3)
    gained   = (filtered_comparison["category"] == "gained").sum()
    lost     = (filtered_comparison["category"] == "lost").sum()
    unchanged= (filtered_comparison["category"] == "unchanged").sum()
    fc1.metric("Players who gained runs",     gained,    delta=f"+{gained}")
    fc2.metric("Players who lost runs",       lost,      delta=f"-{lost}" if lost else "0")
    fc3.metric("Players unchanged",           unchanged)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        # Delta distribution
        fig_delta = px.histogram(
            filtered_comparison,
            x="run_delta",
            color="category",
            color_discrete_map={
                "gained":    GREEN,
                "lost":      RED,
                "unchanged": "#cccccc",
            },
            labels={"run_delta": "Run delta (relative − absolute)", "count": "Players"},
            nbins=30,
        )
        fig_delta.update_layout(**base_layout("Distribution of run delta", height=350))
        fig_delta.update_traces(marker=dict(line=dict(width=0)))
        fig_delta.add_vline(x=0, line=dict(color=AMBER, width=1, dash="dash"))
        st.plotly_chart(fig_delta, use_container_width=True)

    with col_c2:
        # v-max vs threshold relative to 20 km/h
        fig_thresh = px.scatter(
            filtered_comparison,
            x="vmax_kmh",
            y="threshold_kmh",
            color="category",
            color_discrete_map={
                "gained":    GREEN,
                "lost":      RED,
                "unchanged": "#cccccc",
            },
            hover_data=["player_name", "team_name", "run_delta"],
            labels={
                "vmax_kmh": "Personal v-max (km/h)",
                "threshold_at_pct": f"Threshold at {threshold_pct*100:.0f}% (km/h)",
            },
        )
        fig_thresh.add_hline(y=20, line=dict(color=AMBER, width=1, dash="dot"),
                             annotation_text="20 km/h industry standard",
                             annotation_font=dict(size=9, color=AMBER))
        fig_thresh.update_layout(**base_layout("Threshold vs v-max", height=350))
        fig_thresh.update_traces(marker=dict(size=6, line=dict(width=0)))
        st.plotly_chart(fig_thresh, use_container_width=True)

    # Team outperformance
    st.markdown("#### Team outperformance vs industry standard")
    team_comp = (
        filtered_comparison
        .groupby(["team_name", "team_short"])
        .agg(
            total_runs_relative = ("runs_relative", "sum"),
            total_runs_absolute = ("runs_absolute", "sum"),
            total_run_delta     = ("run_delta", "sum"),
            players_gained      = ("category", lambda x: (x == "gained").sum()),
            players_lost        = ("category", lambda x: (x == "lost").sum()),
            n_players           = ("player_name", "count"),
        )
        .reset_index()
    )
    team_comp["pct_outperformance"] = (
        (team_comp["total_runs_relative"] - team_comp["total_runs_absolute"])
        / team_comp["total_runs_absolute"].clip(lower=1) * 100
    ).round(1)
    team_comp = team_comp.sort_values("pct_outperformance", ascending=False)

    fig_outperf = go.Figure(go.Bar(
        x=team_comp["pct_outperformance"],
        y=team_comp["team_short"],
        orientation="h",
        marker=dict(
            color=[GREEN if v >= 0 else RED for v in team_comp["pct_outperformance"]],
            opacity=0.8,
            line=dict(width=0),
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Outperformance: %{x:.1f}%<br>"
            "<extra></extra>"
        ),
    ))
    fig_outperf.add_vline(x=0, line=dict(color=AMBER, width=1))
    fig_outperf.update_layout(
        **base_layout(
            "% more runs under relative definition vs 20 km/h standard",
            height=700,
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
            xaxis=dict(title="% outperformance"),
        )
    )
    st.plotly_chart(fig_outperf, use_container_width=True)

    # Most affected players
    st.markdown("#### Most affected players")
    most_affected = (
        filtered_comparison
        .reindex(columns=["player_name", "team_short", "pos", "pos_detail",
                          "vmax_kmh", "threshold_at_pct",
                          "runs_absolute_dynamic", "runs_relative_dynamic",
                          "run_delta", "pct_change", "category"])
        .sort_values("run_delta", key=abs, ascending=False)
        .reset_index(drop=True)
    )
    st.caption(f"{len(most_affected):,} players · sorted by absolute run delta")
    st.dataframe(
        most_affected.rename(columns={
            "player_name":          "Player",
            "team_short":           "Team",
            "pos":                  "Pos",
            "pos_detail":           "Pos detail",
            "vmax_kmh":             "v-max (km/h)",
            "threshold_at_pct":     "Threshold (km/h)",
            "runs_absolute_dynamic":"HSR Industry Standard",
            "runs_relative_dynamic":"HSR New Definition",
            "run_delta":            "Delta",
            "pct_change":           "% change",
            "category":             "Category",
        }),
        use_container_width=True,
        height=600,
    )

# ══════════════════════════════════════════════
# TAB 5 — Pitch zones
# ══════════════════════════════════════════════
with tab5:
    st.markdown("### HSR runs by pitch zone")
    st.markdown(
        '<p class="caption">CDF coordinate system: x ∈ [−52.5, 52.5] metres from centre</p>',
        unsafe_allow_html=True
    )

    if "start_x" in runs_df.columns:
        runs_df["pitch_zone"] = pd.cut(
            runs_df["start_x"],
            bins=[-53, -17.5, 17.5, 53],
            labels=["Defensive third", "Middle third", "Attacking third"]
        )

        col_z1, col_z2 = st.columns(2)

        with col_z1:
            zone_agg = (
                runs_df
                .groupby("pitch_zone", observed=True)
                .agg(
                    n_runs          = ("duration_sec", "count"),
                    avg_speed       = ("peak_speed_kmh", "mean"),
                    avg_pct_of_vmax = ("pct_of_vmax", "mean"),
                    avg_distance    = ("distance_m", "mean"),
                    avg_duration    = ("duration_sec", "mean"),
                )
                .reset_index()
            )
            zone_agg["avg_pct_of_vmax"] *= 100

            fig_zone = px.bar(
                zone_agg,
                x="pitch_zone",
                y="n_runs",
                color="avg_pct_of_vmax",
                color_continuous_scale=[[0, "#ddeeff"], [1, "#1a4b8c"]],
                text=zone_agg["n_runs"].apply(lambda x: f"{x:,}"),
                labels={
                    "pitch_zone": "Pitch zone",
                    "n_runs": "Number of HSR runs",
                    "avg_pct_of_vmax": "Avg % of v-max",
                },
            )
            fig_zone.update_traces(
                textposition="outside",
                marker=dict(line=dict(width=0)),
            )
            fig_zone.update_layout(**base_layout("HSR runs per pitch zone", height=380))
            st.plotly_chart(fig_zone, use_container_width=True)

        with col_z2:
            # Position breakdown within zones
            if "position" in runs_df.columns:
                zone_pos = (
                    runs_df
                    .groupby(["pitch_zone", "pos"], observed=True)
                    .size()
                    .reset_index(name="n_runs")
                )
                fig_zone_pos = px.bar(
                    zone_pos,
                    x="pitch_zone",
                    y="n_runs",
                    color="pos",
                    barmode="stack",
                    color_discrete_map=POS_COLORS,
                    labels={
                        "pitch_zone": "Pitch zone",
                        "n_runs": "HSR runs",
                        "pos": "Position group",
                    },
                )
                fig_zone_pos.update_layout(
                    **base_layout("HSR runs by zone and position", height=380)
                )
                fig_zone_pos.update_traces(marker=dict(line=dict(width=0)))
                st.plotly_chart(fig_zone_pos, use_container_width=True)

        # ── 9-Zone HSR Heatmap ────────────────────────────────────────────
        st.markdown("#### HSR heatmap — run start positions")

        # Filter to current threshold + sidebar filters
        pitch_runs = qualifying_runs.copy()
        if "team_short" in pitch_runs.columns:
            pitch_runs = pitch_runs[pitch_runs["team_short"].isin(selected_teams)]
        if "pos" in pitch_runs.columns:
            pitch_runs = pitch_runs[pitch_runs["pos"].isin(selected_positions)]
        if pitch_runs.empty:
            pitch_runs = qualifying_runs.copy()

        st.caption(
            f"{len(pitch_runs):,} HSR runs at {threshold_pct*100:.0f}% threshold "
            f"· home team attacks right"
        )

        # Pitch dimensions (CDF metres, origin at centre)
        # x: -52.5 to 52.5  (length 105m)
        # y: -34 to 34       (width 68m)
        # 9 zones: 3 columns (thirds) × 3 rows (left/centre/right channel)
        x_edges = [-52.5, -17.5,  17.5,  52.5]   # 3 thirds
        y_edges = [-34.0,  -11.33, 11.33, 34.0]   # 3 channels

        x_labels = ["Defensive third", "Middle third", "Attacking third"]
        y_labels = ["Right channel", "Central channel", "Left channel"]

        # Count runs in each zone
        import numpy as np
        zone_counts = np.zeros((3, 3), dtype=int)
        zone_avg_speed = np.zeros((3, 3))
        zone_speed_sum = np.zeros((3, 3))

        if "start_x" in pitch_runs.columns and "start_y" in pitch_runs.columns:
            for _, row in pitch_runs.iterrows():
                xi = np.searchsorted(x_edges[1:], row["start_x"])
                yi = np.searchsorted(y_edges[1:], row["start_y"])
                xi = min(xi, 2)
                yi = min(yi, 2)
                zone_counts[yi, xi] += 1
                if "peak_speed_kmh" in pitch_runs.columns:
                    zone_speed_sum[yi, xi] += row["peak_speed_kmh"]

        # Avg speed per zone
        with np.errstate(divide="ignore", invalid="ignore"):
            zone_avg_speed = np.where(
                zone_counts > 0, zone_speed_sum / zone_counts, 0
            )

        total_runs = zone_counts.sum()
        zone_pct = np.where(
            total_runs > 0, zone_counts / total_runs * 100, 0
        )

        # ── Build pitch figure ────────────────────────────────────────────
        fig_pitch = go.Figure()

        PITCH_BG   = "#f8f8f5"
        PITCH_LINE = "#c8c8c4"
        HEAT_LOW   = "#ffffff"
        HEAT_HIGH  = "#1a4b8c"

        # Background
        fig_pitch.add_shape(
            type="rect", x0=-52.5, x1=52.5, y0=-34, y1=34,
            fillcolor=PITCH_BG, line=dict(color=PITCH_LINE, width=1.5), layer="below"
        )

        # Draw zone rectangles — coloured by run count
        max_count = zone_counts.max() if zone_counts.max() > 0 else 1

        for row_i in range(3):
            for col_i in range(3):
                x0 = x_edges[col_i]
                x1 = x_edges[col_i + 1]
                y0 = y_edges[row_i]
                y1 = y_edges[row_i + 1]
                count = int(zone_counts[row_i, col_i])
                pct   = zone_pct[row_i, col_i]
                speed = zone_avg_speed[row_i, col_i]
                intensity = count / max_count

                # Interpolate fill colour: white → dark blue
                r = int(255 + (26  - 255) * intensity)
                g = int(255 + (75  - 255) * intensity)
                b = int(255 + (140 - 255) * intensity)
                fill = f"rgba({r},{g},{b},0.75)"
                text_col = "#ffffff" if intensity > 0.4 else "#333333"

                fig_pitch.add_shape(
                    type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                    fillcolor=fill,
                    line=dict(color=PITCH_LINE, width=0.8),
                    layer="below"
                )

                # Zone label: count + % on two lines
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2

                fig_pitch.add_annotation(
                    x=cx, y=cy + 3,
                    text=f"<b>{count:,}</b>",
                    showarrow=False,
                    font=dict(family="Inter", size=14, color=text_col),
                    xanchor="center", yanchor="middle",
                )
                fig_pitch.add_annotation(
                    x=cx, y=cy - 3,
                    text=f"{pct:.1f}%",
                    showarrow=False,
                    font=dict(family="Inter", size=10, color=text_col),
                    xanchor="center", yanchor="middle",
                )
                if speed > 0:
                    fig_pitch.add_annotation(
                        x=cx, y=cy - 7.5,
                        text=f"avg {speed:.1f} km/h",
                        showarrow=False,
                        font=dict(family="Inter", size=9,
                                  color=text_col if intensity > 0.4 else "#666"),
                        xanchor="center", yanchor="middle",
                    )

        # Pitch markings on top of zones
        PITCH_LINE_TOP = "#aaaaaa"
        # Halfway line
        fig_pitch.add_shape(type="line", x0=0, x1=0, y0=-34, y1=34,
                            line=dict(color=PITCH_LINE_TOP, width=1.2))
        # Third boundaries (already zone edges but draw explicitly)
        for xb in [-17.5, 17.5]:
            fig_pitch.add_shape(type="line", x0=xb, x1=xb, y0=-34, y1=34,
                                line=dict(color=PITCH_LINE_TOP, width=0.8, dash="dot"))
        # Centre circle
        fig_pitch.add_shape(type="circle", x0=-9.15, x1=9.15, y0=-9.15, y1=9.15,
                            line=dict(color=PITCH_LINE_TOP, width=1))
        # Penalty areas
        fig_pitch.add_shape(type="rect", x0=-52.5, x1=-36, y0=-20.16, y1=20.16,
                            line=dict(color=PITCH_LINE_TOP, width=1))
        fig_pitch.add_shape(type="rect", x0=36, x1=52.5, y0=-20.16, y1=20.16,
                            line=dict(color=PITCH_LINE_TOP, width=1))
        # Outer boundary on top
        fig_pitch.add_shape(type="rect", x0=-52.5, x1=52.5, y0=-34, y1=34,
                            line=dict(color=PITCH_LINE_TOP, width=1.5))

        # Third labels above pitch
        for i, label in enumerate(x_labels):
            cx = (x_edges[i] + x_edges[i+1]) / 2
            fig_pitch.add_annotation(
                x=cx, y=36.5,
                text=label.upper(),
                showarrow=False,
                font=dict(family="Inter", size=10, color="#888"),
                xanchor="center",
            )

        # Channel labels left of pitch
        for i, label in enumerate(y_labels):
            cy = (y_edges[i] + y_edges[i+1]) / 2
            fig_pitch.add_annotation(
                x=-55, y=cy,
                text=label,
                showarrow=False,
                font=dict(family="Inter", size=9, color="#888"),
                xanchor="right",
            )

        # Attack direction arrow
        fig_pitch.add_annotation(
            x=50, y=-36.5,
            ax=38, ay=-36.5,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor="#888",
            text="Attack",
            font=dict(family="Inter", size=9, color="#888"),
            xanchor="right",
        )

        fig_pitch.update_layout(
            plot_bgcolor=PITCH_BG,
            paper_bgcolor=PAPER_BG,
            height=480,
            margin=dict(l=80, r=20, t=30, b=20),
            xaxis=dict(
                range=[-58, 56], showgrid=False, zeroline=False,
                showticklabels=False, showline=False,
            ),
            yaxis=dict(
                range=[-40, 40], showgrid=False, zeroline=False,
                showticklabels=False, showline=False,
                scaleanchor="x", scaleratio=1,
            ),
            showlegend=False,
        )

        st.plotly_chart(fig_pitch, use_container_width=True)


    else:
        st.info("Pitch zone analysis requires start_x/start_y columns in hsr_runs.csv")

# ══════════════════════════════════════════════
# TAB 6 — Match analysis
# ══════════════════════════════════════════════
with tab6:
    st.markdown("### Match analysis")

    if match_meta_df.empty:
        st.info("Add `match_metadata.csv` to the `data/` folder to enable match analysis.")
    else:
        # ── Match selector ────────────────────────────────────────────────
        match_meta_df["label"] = (
            match_meta_df["home_team_short"] + " vs " +
            match_meta_df["away_team_short"] + "  ·  W" +
            match_meta_df["week"].astype(str) +
            "  (" + match_meta_df["Round"].str[:2].str.upper() + ")"
        )
        match_options = match_meta_df.sort_values("date")["label"].tolist()

        # Default to the game selected in sidebar if one is chosen
        default_idx = 0
        if selected_game_id:
            sid = str(selected_game_id)
            match = match_meta_df[match_meta_df["game_id"].astype(str) == sid]
            if not match.empty:
                lbl = match["label"].iloc[0]
                if lbl in match_options:
                    default_idx = match_options.index(lbl)

        selected_match_label = st.selectbox(
            "Select match",
            options=match_options,
            index=default_idx,
            key="match_tab_selector"
        )

        match_row = match_meta_df[match_meta_df["label"] == selected_match_label].iloc[0]
        gid = str(match_row["game_id"])

        # ── Match header ─────────────────────────────────────────────────
        home = match_row["home_team_name"]
        away = match_row["away_team_name"]
        date = match_row["date"][:10]
        stadium = match_row["stadium_name"]
        rnd  = match_row["Round"]
        week = match_row["week"]

        c1, c2, c3 = st.columns([2, 1, 2])
        c1.markdown(
            f'<div style="text-align:right;">'
            f'<p class="eyebrow">Home</p>'
            f'<p style="font-family:Georgia,serif;font-size:24px;'
            f'font-weight:600;color:#111;margin:0;">{home}</p></div>',
            unsafe_allow_html=True
        )
        c2.markdown(
            f'<div style="text-align:center;padding-top:20px;">'
            f'<p style="font-family:JetBrains Mono,monospace;font-size:13px;'
            f'color:#888;margin:0;">{date}</p>'
            f'<p style="font-family:Inter;font-size:11px;color:#aaa;margin:2px 0;">'
            f'{rnd} · W{week}</p></div>',
            unsafe_allow_html=True
        )
        c3.markdown(
            f'<div style="text-align:left;">'
            f'<p class="eyebrow">Away</p>'
            f'<p style="font-family:Georgia,serif;font-size:24px;'
            f'font-weight:600;color:#111;margin:0;">{away}</p></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<p style="text-align:center;font-family:Inter;font-size:12px;'
            f'color:#888;margin:4px 0 16px 0;">{stadium}</p>',
            unsafe_allow_html=True
        )
        st.divider()

        # ── Filter data to this match ─────────────────────────────────────
        match_runs = qualifying_runs[qualifying_runs["game_id"].astype(str) == gid].copy()

        if match_runs.empty:
            # Fall back to unfiltered runs for this game
            match_runs = runs_df[runs_df["game_id"].astype(str) == gid].copy()
            match_runs = match_runs[match_runs["pct_of_vmax"] >= threshold_pct].copy()

        if match_runs.empty:
            st.warning("No HSR runs found for this match at the current threshold. Try lowering the threshold.")
        else:
            st.caption(f"{len(match_runs):,} HSR runs in this match at {threshold_pct*100:.0f}% threshold")

            # ── Top performers ───────────────────────────────────────────
            st.markdown("#### Top performers")

            if "player_name" in match_runs.columns:
                top_perf = (
                    match_runs
                    .groupby(["player_id", "player_name", "team_short", "pos"])
                    .agg(
                        runs        = ("run_id", "count"),
                        total_dist  = ("distance_m", "sum"),
                        peak_speed  = ("peak_speed_kmh", "max"),
                        avg_speed   = ("peak_speed_kmh", "mean"),
                        avg_pct_vmax= ("pct_of_vmax", "mean"),
                    )
                    .reset_index()
                    .sort_values("runs", ascending=False)
                )
                top_perf["avg_pct_vmax"] = (top_perf["avg_pct_vmax"] * 100).round(1)
                top_perf["avg_speed"]    = top_perf["avg_speed"].round(1)
                top_perf["peak_speed"]   = top_perf["peak_speed"].round(1)
                top_perf["total_dist"]   = top_perf["total_dist"].round(0)

                mp1, mp2 = st.columns(2)

                with mp1:
                    fig_top = go.Figure(go.Bar(
                        x=top_perf["runs"].head(15),
                        y=(top_perf["player_name"] + " (" + top_perf["team_short"] + ")").head(15),
                        orientation="h",
                        marker=dict(
                            color=[POS_COLORS.get(p, "#aaa") for p in top_perf["pos"].head(15)],
                            opacity=0.85, line=dict(width=0),
                        ),
                        hovertemplate="<b>%{y}</b><br>HSR runs: %{x}<extra></extra>",
                    ))
                    fig_top.update_layout(
                        **base_layout(
                            "HSR runs in match",
                            height=420,
                            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                            xaxis=dict(title="HSR runs"),
                        )
                    )
                    st.plotly_chart(fig_top, use_container_width=True)

                with mp2:
                    st.dataframe(
                        top_perf
                        .rename(columns={
                            "player_name":  "Player",
                            "team_short":   "Team",
                            "pos":          "Pos",
                            "runs":         "HSR runs",
                            "total_dist":   "Total dist (m)",
                            "peak_speed":   "Peak speed",
                            "avg_speed":    "Avg speed",
                            "avg_pct_vmax": "Avg % v-max",
                        })
                        .drop(columns=["player_id"])
                        .reset_index(drop=True),
                        use_container_width=True,
                        height=420,
                    )

            # ── Team comparison ──────────────────────────────────────────
            st.markdown("#### Team comparison")

            if "team_short" in match_runs.columns:
                team_match = (
                    match_runs
                    .groupby("team_short")
                    .agg(
                        total_runs  = ("run_id", "count"),
                        total_dist  = ("distance_m", "sum"),
                        avg_speed   = ("peak_speed_kmh", "mean"),
                        peak_speed  = ("peak_speed_kmh", "max"),
                        avg_pct_vmax= ("pct_of_vmax", "mean"),
                        n_players   = ("player_id", "nunique"),
                    )
                    .reset_index()
                )
                team_match["avg_pct_vmax"] = (team_match["avg_pct_vmax"] * 100).round(1)

                tm1, tm2, tm3 = st.columns(3)
                for col, metric, label in [
                    (tm1, "total_runs",  "Total HSR runs"),
                    (tm2, "total_dist",  "Total HSR distance (m)"),
                    (tm3, "avg_speed",   "Avg peak speed (km/h)"),
                ]:
                    fig_t = go.Figure(go.Bar(
                        x=team_match["team_short"],
                        y=team_match[metric].round(1),
                        marker=dict(color=[BLUE, RED][:len(team_match)],
                                    opacity=0.8, line=dict(width=0)),
                        text=team_match[metric].round(1),
                        textposition="outside",
                    ))
                    fig_t.update_layout(**base_layout(label, height=280))
                    col.plotly_chart(fig_t, use_container_width=True)

            # ── Position breakdown ───────────────────────────────────────
            st.markdown("#### HSR by position")

            if "pos" in match_runs.columns:
                pos_match = (
                    match_runs
                    .groupby("pos")
                    .agg(
                        total_runs  = ("run_id", "count"),
                        avg_pct_vmax= ("pct_of_vmax", "mean"),
                        avg_speed   = ("peak_speed_kmh", "mean"),
                    )
                    .reset_index()
                    .sort_values("total_runs", ascending=False)
                )
                pos_match["avg_pct_vmax"] = (pos_match["avg_pct_vmax"] * 100).round(1)

                fig_pos_m = px.bar(
                    pos_match,
                    x="pos", y="total_runs",
                    color="pos",
                    color_discrete_map=POS_COLORS,
                    text="total_runs",
                    labels={"pos": "Position", "total_runs": "HSR runs"},
                )
                fig_pos_m.update_traces(marker=dict(line=dict(width=0)),
                                        textposition="outside")
                fig_pos_m.update_layout(**base_layout("HSR runs by position", height=320))
                st.plotly_chart(fig_pos_m, use_container_width=True)

            # ── Pitch heatmap ────────────────────────────────────────────
            st.markdown("#### Pitch heatmap")

            if "start_x" in match_runs.columns and "start_y" in match_runs.columns:
                x_edges = [-52.5, -17.5, 17.5, 52.5]
                y_edges = [-34.0, -11.33, 11.33, 34.0]
                x_labels = ["Defensive third", "Middle third", "Attacking third"]
                y_labels  = ["Right channel", "Central channel", "Left channel"]

                zone_counts   = np.zeros((3, 3), dtype=int)
                zone_speed_sum= np.zeros((3, 3))

                for _, row in match_runs.iterrows():
                    xi = min(np.searchsorted(x_edges[1:], row["start_x"]), 2)
                    yi = min(np.searchsorted(y_edges[1:], row["start_y"]), 2)
                    zone_counts[yi, xi] += 1
                    if "peak_speed_kmh" in match_runs.columns:
                        zone_speed_sum[yi, xi] += row["peak_speed_kmh"]

                with np.errstate(divide="ignore", invalid="ignore"):
                    zone_avg_speed = np.where(zone_counts > 0,
                                              zone_speed_sum / zone_counts, 0)
                total_r  = zone_counts.sum()
                zone_pct_m = np.where(total_r > 0, zone_counts / total_r * 100, 0)
                max_count  = zone_counts.max() if zone_counts.max() > 0 else 1

                fig_pm = go.Figure()
                PITCH_BG   = "#f8f8f5"
                PITCH_LINE = "#c8c8c4"

                fig_pm.add_shape(
                    type="rect", x0=-52.5, x1=52.5, y0=-34, y1=34,
                    fillcolor=PITCH_BG, line=dict(color=PITCH_LINE, width=1.5), layer="below"
                )
                for row_i in range(3):
                    for col_i in range(3):
                        x0, x1 = x_edges[col_i], x_edges[col_i+1]
                        y0, y1 = y_edges[row_i], y_edges[row_i+1]
                        count  = int(zone_counts[row_i, col_i])
                        pct    = zone_pct_m[row_i, col_i]
                        speed  = zone_avg_speed[row_i, col_i]
                        inten  = count / max_count
                        r = int(255 + (26  - 255) * inten)
                        g = int(255 + (75  - 255) * inten)
                        b = int(255 + (140 - 255) * inten)
                        fill = f"rgba({r},{g},{b},0.75)"
                        tcol = "#ffffff" if inten > 0.4 else "#333333"
                        cx, cy = (x0+x1)/2, (y0+y1)/2
                        fig_pm.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                                         fillcolor=fill, line=dict(color=PITCH_LINE, width=0.8), layer="below")
                        fig_pm.add_annotation(x=cx, y=cy+3, text=f"<b>{count}</b>",
                                              showarrow=False, font=dict(family="Inter", size=14, color=tcol),
                                              xanchor="center", yanchor="middle")
                        fig_pm.add_annotation(x=cx, y=cy-3, text=f"{pct:.1f}%",
                                              showarrow=False, font=dict(family="Inter", size=10, color=tcol),
                                              xanchor="center", yanchor="middle")
                        if speed > 0:
                            fig_pm.add_annotation(x=cx, y=cy-7.5, text=f"avg {speed:.1f} km/h",
                                                  showarrow=False,
                                                  font=dict(family="Inter", size=9, color=tcol if inten > 0.4 else "#666"),
                                                  xanchor="center", yanchor="middle")

                PITCH_LINE_TOP = "#aaaaaa"
                for sh in [
                    dict(type="line", x0=0, x1=0, y0=-34, y1=34, line=dict(color=PITCH_LINE_TOP, width=1.2)),
                    dict(type="circle", x0=-9.15, x1=9.15, y0=-9.15, y1=9.15, line=dict(color=PITCH_LINE_TOP, width=1)),
                    dict(type="rect", x0=-52.5, x1=-36, y0=-20.16, y1=20.16, line=dict(color=PITCH_LINE_TOP, width=1)),
                    dict(type="rect", x0=36, x1=52.5, y0=-20.16, y1=20.16, line=dict(color=PITCH_LINE_TOP, width=1)),
                    dict(type="rect", x0=-52.5, x1=52.5, y0=-34, y1=34, line=dict(color=PITCH_LINE_TOP, width=1.5)),
                ]:
                    fig_pm.add_shape(**sh)
                for xb in [-17.5, 17.5]:
                    fig_pm.add_shape(type="line", x0=xb, x1=xb, y0=-34, y1=34,
                                     line=dict(color=PITCH_LINE_TOP, width=0.8, dash="dot"))

                for i, lbl in enumerate(x_labels):
                    fig_pm.add_annotation(x=(x_edges[i]+x_edges[i+1])/2, y=36.5,
                                          text=lbl.upper(), showarrow=False,
                                          font=dict(family="Inter", size=10, color="#888"), xanchor="center")
                for i, lbl in enumerate(y_labels):
                    fig_pm.add_annotation(x=-55, y=(y_edges[i]+y_edges[i+1])/2,
                                          text=lbl, showarrow=False,
                                          font=dict(family="Inter", size=9, color="#888"), xanchor="right")

                fig_pm.update_layout(
                    plot_bgcolor=PITCH_BG, paper_bgcolor=PAPER_BG,
                    height=460,
                    margin=dict(l=80, r=20, t=30, b=20),
                    xaxis=dict(range=[-58, 56], showgrid=False, zeroline=False,
                               showticklabels=False, showline=False),
                    yaxis=dict(range=[-40, 40], showgrid=False, zeroline=False,
                               showticklabels=False, showline=False,
                               scaleanchor="x", scaleratio=1),
                    showlegend=False,
                )
                st.plotly_chart(fig_pm, use_container_width=True)



# ══════════════════════════════════════════════
# TAB 7 — Tournament Phases
# ══════════════════════════════════════════════
with tab7:
    st.markdown("### Group Stage vs Knockout")
    st.markdown(
        '<p class="kicker">How does high-speed running change as the tournament progresses '
        'and the stakes increase?</p>',
        unsafe_allow_html=True
    )
    st.divider()

    if match_meta_df.empty:
        st.info("Add `match_metadata.csv` to the `data/` folder to enable this tab.")
    elif "game_id" not in runs_df.columns:
        st.info("hsr_runs.csv requires a game_id column.")
    else:
        runs_with_round = runs_df.merge(
            match_meta_df[["game_id", "Round"]].assign(
                game_id=match_meta_df["game_id"].astype(str)
            ),
            left_on=runs_df["game_id"].astype(str),
            right_on="game_id",
            how="left"
        )
        runs_with_round = runs_with_round[
            runs_with_round["pct_of_vmax"] >= threshold_pct
        ]

        # Apply team/position filters
        if "team_short" in runs_with_round.columns:
            runs_with_round = runs_with_round[
                runs_with_round["team_short"].isin(selected_teams)
            ]
        if "pos" in runs_with_round.columns:
            runs_with_round = runs_with_round[
                runs_with_round["pos"].isin(selected_positions)
            ]

        if runs_with_round.empty:
            st.warning("No runs match the current filters.")
        else:
            round_agg = (
                runs_with_round
                .groupby("Round")
                .agg(
                    total_runs      = ("run_id", "count"),
                    avg_pct_vmax    = ("pct_of_vmax", "mean"),
                    avg_peak_speed  = ("peak_speed_kmh", "mean"),
                    avg_distance    = ("distance_m", "mean"),
                    avg_duration    = ("duration_sec", "mean"),
                )
                .reset_index()
            )
            round_agg["avg_pct_vmax"] = (round_agg["avg_pct_vmax"] * 100).round(1)

            n_gs = match_meta_df[match_meta_df["Round"] == "Group Stage"].shape[0]
            n_ko = match_meta_df[match_meta_df["Round"] == "Knockout"].shape[0]
            round_agg["n_games"] = round_agg["Round"].map(
                {"Group Stage": n_gs, "Knockout": n_ko}
            )
            round_agg["runs_per_game"] = (
                round_agg["total_runs"] / round_agg["n_games"]
            ).round(1)

            rnd_colors = {"Group Stage": BLUE, "Knockout": RED}

            # ── KPI row ───────────────────────────────────────────────────
            st.markdown("#### Summary metrics")
            kpi_cols = st.columns(4)
            metrics_kpi = [
                ("runs_per_game",  "Runs per game"),
                ("avg_pct_vmax",   "Avg intensity (% v-max)"),
                ("avg_peak_speed", "Avg peak speed (km/h)"),
                ("avg_distance",   "Avg run distance (m)"),
            ]
            for col, (metric, label) in zip(kpi_cols, metrics_kpi):
                fig_r = go.Figure(go.Bar(
                    x=round_agg["Round"],
                    y=round_agg[metric].round(1),
                    marker=dict(
                        color=[rnd_colors.get(r, BLUE) for r in round_agg["Round"]],
                        opacity=0.85, line=dict(width=0),
                    ),
                    text=round_agg[metric].round(1),
                    textposition="outside",
                ))
                fig_r.update_layout(**base_layout(label, height=300))
                col.plotly_chart(fig_r, use_container_width=True)

            st.divider()

            # ── Week-by-week trend ────────────────────────────────────────
            st.markdown("#### Week-by-week trend")
            st.markdown(
                '<p style="font-family:Inter;font-size:12px;color:#666;">'
                'Average HSR runs per player per game across tournament weeks.</p>',
                unsafe_allow_html=True
            )

            week_meta = match_meta_df[["game_id", "week", "Round"]].assign(
                game_id=match_meta_df["game_id"].astype(str)
            )
            runs_with_week = runs_df.merge(
                week_meta,
                left_on=runs_df["game_id"].astype(str),
                right_on="game_id",
                how="left"
            )
            runs_with_week = runs_with_week[
                runs_with_week["pct_of_vmax"] >= threshold_pct
            ]

            week_agg = (
                runs_with_week
                .groupby(["week_y" if "week_y" in runs_with_week.columns else "week", "Round"])
                .agg(
                    total_runs      = ("run_id", "count"),
                    n_games         = ("game_id_y" if "game_id_y" in runs_with_week.columns
                                       else "game_id", "nunique"),
                    avg_peak_speed  = ("peak_speed_kmh", "mean"),
                    avg_pct_vmax    = ("pct_of_vmax", "mean"),
                )
                .reset_index()
            )
            week_col = "week_y" if "week_y" in week_agg.columns else "week"
            week_agg["runs_per_game"] = (
                week_agg["total_runs"] / week_agg["n_games"].clip(lower=1)
            ).round(1)
            week_agg["avg_pct_vmax"] = (week_agg["avg_pct_vmax"] * 100).round(1)
            week_agg = week_agg.sort_values(week_col)

            tw1, tw2 = st.columns(2)

            with tw1:
                fig_week = go.Figure()
                for rnd, color in rnd_colors.items():
                    sub = week_agg[week_agg["Round"] == rnd]
                    if sub.empty:
                        continue
                    fig_week.add_trace(go.Scatter(
                        x=sub[week_col],
                        y=sub["runs_per_game"],
                        mode="lines+markers",
                        name=rnd,
                        line=dict(color=color, width=2),
                        marker=dict(size=8, color=color),
                        hovertemplate=f"<b>{rnd}</b><br>Week %{{x}}<br>Runs/game: %{{y}}<extra></extra>",
                    ))
                fig_week.update_layout(
                    **base_layout(
                        "HSR runs per game by week",
                        height=340,
                        xaxis=dict(title="Tournament week", dtick=1),
                        yaxis=dict(title="Runs per game"),
                    )
                )
                st.plotly_chart(fig_week, use_container_width=True)

            with tw2:
                fig_intensity = go.Figure()
                for rnd, color in rnd_colors.items():
                    sub = week_agg[week_agg["Round"] == rnd]
                    if sub.empty:
                        continue
                    fig_intensity.add_trace(go.Scatter(
                        x=sub[week_col],
                        y=sub["avg_pct_vmax"],
                        mode="lines+markers",
                        name=rnd,
                        line=dict(color=color, width=2),
                        marker=dict(size=8, color=color),
                        hovertemplate=f"<b>{rnd}</b><br>Week %{{x}}<br>Avg % v-max: %{{y}}<extra></extra>",
                    ))
                fig_intensity.update_layout(
                    **base_layout(
                        "Avg intensity (% v-max) by week",
                        height=340,
                        xaxis=dict(title="Tournament week", dtick=1),
                        yaxis=dict(title="Avg % of v-max"),
                    )
                )
                st.plotly_chart(fig_intensity, use_container_width=True)

            st.divider()

            # ── Position breakdown by round ───────────────────────────────
            st.markdown("#### Position breakdown by round")

            if "pos" in runs_with_round.columns:
                pos_round = (
                    runs_with_round
                    .groupby(["Round", "pos"])
                    .agg(total_runs=("run_id", "count"))
                    .reset_index()
                )
                games_by_round = match_meta_df.groupby("Round").size().reset_index(name="n_games")
                pos_round = pos_round.merge(games_by_round, on="Round", how="left")
                pos_round["runs_per_game"] = (
                    pos_round["total_runs"] / pos_round["n_games"]
                ).round(2)

                fig_pos_round = px.bar(
                    pos_round,
                    x="pos", y="runs_per_game",
                    color="Round",
                    barmode="group",
                    color_discrete_map={"Group Stage": BLUE, "Knockout": RED},
                    labels={
                        "pos": "Position",
                        "runs_per_game": "HSR runs per game",
                        "Round": "Round",
                    },
                    text=pos_round["runs_per_game"],
                )
                fig_pos_round.update_traces(
                    texttemplate="%{text:.1f}",
                    textposition="outside",
                    marker=dict(line=dict(width=0)),
                )
                fig_pos_round.update_layout(
                    **base_layout("HSR runs per game by position and round", height=380)
                )
                st.plotly_chart(fig_pos_round, use_container_width=True)

            st.divider()

            # ── Summary table ─────────────────────────────────────────────
            st.markdown("#### Summary table")
            st.dataframe(
                round_agg
                .rename(columns={
                    "Round":          "Round",
                    "n_games":        "Games",
                    "total_runs":     "Total runs",
                    "runs_per_game":  "Runs / game",
                    "avg_pct_vmax":   "Avg intensity (% v-max)",
                    "avg_peak_speed": "Avg peak speed (km/h)",
                    "avg_distance":   "Avg distance (m)",
                    "avg_duration":   "Avg duration (s)",
                })
                .round(2),
                use_container_width=True,
                hide_index=True,
            )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center;font-family:Inter,sans-serif;font-size:11px;color:#aaa;">'
    'GradientSports FIFA World Cup 2022 · fast-forward-football · Azure Databricks Delta Lake · '
    'Built for USSF Data Scientist application</p>',
    unsafe_allow_html=True
)
