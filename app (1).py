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
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Main background */
.stApp {
    background-color: #0a0e1a;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f1525;
    border-right: 1px solid #1e2a45;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #1e2a45;
    border-radius: 12px;
    padding: 16px;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    letter-spacing: -0.02em;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2a45;
    border-radius: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #111827;
    border-radius: 8px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #6b7897;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    border-radius: 6px;
}

.stTabs [aria-selected="true"] {
    background-color: #1e2a45 !important;
    color: #4fc3f7 !important;
}

/* Slider */
[data-testid="stSlider"] > div {
    padding-top: 8px;
}

/* Divider */
hr {
    border-color: #1e2a45;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background-color: #111827;
    border-color: #1e2a45;
}

/* Caption text */
.caption {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.highlight {
    color: #4fc3f7;
    font-weight: 700;
}

.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-family: 'DM Mono', monospace;
}

.pill-gained { background-color: #0d3320; color: #4ade80; border: 1px solid #166534; }
.pill-lost   { background-color: #3b0a0a; color: #f87171; border: 1px solid #7f1d1d; }
.pill-unch   { background-color: #1a1f2e; color: #94a3b8; border: 1px solid #334155; }
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
    return summary, comparison, runs

try:
    summary_df, comparison_df, runs_df = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ HSR Metric")
    st.markdown(
        '<p class="caption">FIFA World Cup 2022 · GradientSports tracking</p>',
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown("### Threshold")
    st.markdown(
        '<p class="caption">% of personal v-max to qualify as high-speed run</p>',
        unsafe_allow_html=True
    )

    threshold_pct = st.slider(
        label="Relative threshold",
        min_value=0.60,
        max_value=0.95,
        value=0.75,
        step=0.05,
        format="%.0f%%",
        help="75% = industry-comparable. Raise to see only near-maximum efforts."
    )

    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1e2a45;border-radius:8px;padding:12px;margin-top:8px;">
        <p class="caption" style="margin:0 0 6px 0;">Current threshold</p>
        <p style="font-size:28px;font-weight:800;color:#4fc3f7;margin:0;font-family:'Syne',sans-serif;">
            {threshold_pct*100:.0f}%
        </p>
        <p style="font-size:12px;color:#6b7897;margin:4px 0 0 0;font-family:'DM Mono',monospace;">
            of personal v-max
        </p>
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

        positions = sorted(summary_df["position"].dropna().unique().tolist())
        selected_positions = st.multiselect(
            "Positions",
            options=positions,
            default=positions,
            placeholder="All positions"
        )

        min_games = st.slider("Min games played", 1, 7, 2)

    st.divider()
    st.markdown(
        '<p class="caption">Definition: a run where a player reaches ≥ threshold% '
        'of their personal v-max and maintains it for ≥ 1 second.</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="caption">Industry standard: flat 20 km/h threshold.</p>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown(
        f'<h1 style="margin-bottom:4px;">Relative High-Speed Running</h1>'
        f'<p style="color:#6b7897;font-family:\'DM Mono\',monospace;font-size:13px;margin:0;">'
        f'FIFA Men\'s World Cup 2022 · GradientSports broadcast tracking · '
        f'Threshold: <span style="color:#4fc3f7;">{threshold_pct*100:.0f}% of v-max</span></p>',
        unsafe_allow_html=True
    )
with col_badge:
    st.markdown(
        '<div style="text-align:right;padding-top:12px;">'
        '<span style="background:#0d2137;color:#4fc3f7;border:1px solid #1e4a6e;'
        'padding:6px 14px;border-radius:6px;font-family:\'DM Mono\',monospace;font-size:12px;">'
        '64 games · 32 teams</span></div>',
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
# Recompute metric at selected threshold
# ─────────────────────────────────────────────
# Re-derive runs_per_game and threshold at selected pct
# We have vmax_kmh in summary — recompute threshold and flag qualifying players
summary_df["threshold_at_pct"] = summary_df["vmax_kmh"] * threshold_pct
summary_df["qualifies_relative"] = (
    summary_df["vmax_kmh"] * threshold_pct
) <= summary_df["tournament_peak_speed_kmh"]

# Recompute comparison metrics at new threshold
comparison_df["threshold_at_pct"] = comparison_df["vmax_kmh"] * threshold_pct
comparison_df["above_absolute"]   = comparison_df["threshold_kmh"] <= 20.0
comparison_df["new_threshold_vs_20"] = comparison_df["threshold_at_pct"] - 20.0

# Apply sidebar filters
filtered_summary = summary_df[
    summary_df["team_short"].isin(selected_teams) &
    summary_df["position"].isin(selected_positions) &
    (summary_df["games_appeared"] >= min_games) &
    (~summary_df["low_confidence"])
].copy()

filtered_comparison = comparison_df[
    comparison_df["team_short"].isin(selected_teams) &
    comparison_df["position"].isin(selected_positions)
].copy()

# ─────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

n_below_20 = (comparison_df["threshold_at_pct"] < 20.0).sum()
pct_below  = n_below_20 / len(comparison_df) * 100
avg_vmax   = filtered_summary["vmax_kmh"].mean()
avg_runs   = filtered_summary["runs_per_game"].mean()
top_speed  = filtered_summary["tournament_peak_speed_kmh"].max()
n_players  = len(filtered_summary)

k1.metric("Players tracked",       f"{n_players:,}")
k2.metric("Avg v-max",             f"{avg_vmax:.1f} km/h")
k3.metric("Avg HSR runs / game",   f"{avg_runs:.1f}")
k4.metric("Top speed recorded",    f"{top_speed:.1f} km/h")
k5.metric("Players below 20 km/h threshold", f"{n_below_20} ({pct_below:.0f}%)",
          help="Players whose relative threshold sits below 20 km/h — "
               "meaning the industry standard misses their high-effort runs")

st.divider()

# ─────────────────────────────────────────────
# Plotly theme
# ─────────────────────────────────────────────
PLOT_BG    = "#0a0e1a"
PAPER_BG   = "#0a0e1a"
GRID_COLOR = "#1e2a45"
TEXT_COLOR = "#94a3b8"
ACCENT     = "#4fc3f7"
GREEN      = "#4ade80"
RED        = "#f87171"
AMBER      = "#fbbf24"

def base_layout(title="", height=400, xaxis=None, yaxis=None):
    default_axis = dict(gridcolor=GRID_COLOR, showline=False, zeroline=False)
    x = {**default_axis, **(xaxis or {})}
    y = {**default_axis, **(yaxis or {})}
    return dict(
        title=dict(text=title, font=dict(family="Syne", size=14, color="#e8eaf0")),
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Player ranking",
    "🏟️  Team analysis",
    "🎯  Position analysis",
    "⚖️  Definition comparison",
    "🗺️  Pitch zones",
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
        top_n = st.select_slider("Show top N players", options=[10, 15, 20, 25, 30], value=20)
        top_players = filtered_summary.nlargest(top_n, "runs_per_game")

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=top_players["runs_per_game"],
            y=top_players["player_name"] + " (" + top_players["team_short"] + ")",
            orientation="h",
            marker=dict(
                color=top_players["mean_pct_of_vmax_pct"],
                colorscale=[[0, "#1e3a5f"], [0.5, "#2196f3"], [1.0, "#4fc3f7"]],
                colorbar=dict(
                    title=dict(text="Avg % of v-max", font=dict(size=10)),
                    thickness=10, len=0.6,
                ),
                line=dict(width=0),
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
            y="runs_per_game",
            color="position",
            size="games_appeared",
            hover_data=["player_name", "team_name", "tournament_peak_speed_kmh"],
            labels={
                "vmax_kmh": "Personal v-max (km/h)",
                "runs_per_game": "HSR runs per game",
                "position": "Position",
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

    st.divider()
    st.markdown("#### Full player table")
    display_cols = [
        "player_name", "team_name", "position", "games_appeared",
        "vmax_kmh", "threshold_kmh", "total_runs", "runs_per_game",
        "hsr_distance_per_game_m", "mean_pct_of_vmax_pct",
        "tournament_peak_speed_kmh",
    ]
    st.dataframe(
        filtered_summary[display_cols]
        .sort_values("runs_per_game", ascending=False)
        .reset_index(drop=True)
        .rename(columns={
            "player_name": "Player", "team_name": "Team",
            "position": "Pos", "games_appeared": "Games",
            "vmax_kmh": "v-max", "threshold_kmh": "Threshold",
            "total_runs": "Total runs", "runs_per_game": "Runs/game",
            "hsr_distance_per_game_m": "HSR dist/game (m)",
            "mean_pct_of_vmax_pct": "Avg % v-max",
            "tournament_peak_speed_kmh": "Peak speed",
        }),
        use_container_width=True,
        height=350,
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
            avg_runs_per_game       = ("runs_per_game", "mean"),
            total_team_runs         = ("total_runs", "sum"),
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
            color_continuous_scale=[[0, "#1e3a5f"], [1, "#4fc3f7"]],
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

# ══════════════════════════════════════════════
# TAB 3 — Position analysis
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### HSR by position")

    pos_agg = (
        filtered_summary
        .groupby("position")
        .agg(
            avg_runs_per_game   = ("runs_per_game", "mean"),
            avg_vmax            = ("vmax_kmh", "mean"),
            avg_threshold       = ("threshold_kmh", "mean"),
            avg_distance        = ("hsr_distance_per_game_m", "mean"),
            avg_intensity       = ("mean_pct_of_vmax_pct", "mean"),
            avg_peak_speed      = ("tournament_peak_speed_kmh", "mean"),
            n_players           = ("player_name", "count"),
        )
        .reset_index()
        .sort_values("avg_runs_per_game", ascending=False)
    )

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        fig_pos = px.bar(
            pos_agg,
            x="position",
            y="avg_runs_per_game",
            color="avg_intensity",
            color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#2196f3"], [1, "#4fc3f7"]],
            text=pos_agg["avg_runs_per_game"].round(1),
            labels={
                "position": "Position",
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
        colors    = ["#4fc3f7", "#4ade80", "#fbbf24", "#f87171",
                     "#a78bfa", "#fb923c", "#34d399"]

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
                name=row["position"],
                line=dict(color=colors[i % len(colors)], width=2),
                fill="toself",
                fillcolor=colors[i % len(colors)],
                opacity=0.15,
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor=PLOT_BG,
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
            "position": "Position",
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
                "unchanged": "#475569",
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
                "unchanged": "#475569",
            },
            hover_data=["player_name", "team_name", "run_delta"],
            labels={
                "vmax_kmh": "Personal v-max (km/h)",
                "threshold_kmh": f"Threshold at {threshold_pct*100:.0f}% (km/h)",
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
            color=team_comp["pct_outperformance"],
            colorscale=[[0, RED], [0.5, "#475569"], [1.0, GREEN]],
            cmid=0,
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
        .reindex(columns=["player_name", "team_short", "position",
                          "vmax_kmh", "threshold_kmh",
                          "runs_absolute", "runs_relative",
                          "run_delta", "pct_change", "category"])
        .sort_values("run_delta", key=abs, ascending=False)
        .head(30)
        .reset_index(drop=True)
    )
    st.dataframe(
        most_affected.rename(columns={
            "player_name": "Player", "team_short": "Team",
            "position": "Pos", "vmax_kmh": "v-max",
            "threshold_kmh": "Threshold", "runs_absolute": "Runs (20 km/h)",
            "runs_relative": "Runs (relative)", "run_delta": "Delta",
            "pct_change": "% change", "category": "Category",
        }),
        use_container_width=True,
        height=400,
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
                color_continuous_scale=[[0, "#1e3a5f"], [1, "#4fc3f7"]],
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
                    .groupby(["pitch_zone", "position"], observed=True)
                    .size()
                    .reset_index(name="n_runs")
                )
                fig_zone_pos = px.bar(
                    zone_pos,
                    x="pitch_zone",
                    y="n_runs",
                    color="position",
                    barmode="stack",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    labels={
                        "pitch_zone": "Pitch zone",
                        "n_runs": "HSR runs",
                        "position": "Position",
                    },
                )
                fig_zone_pos.update_layout(
                    **base_layout("HSR runs by zone and position", height=380)
                )
                fig_zone_pos.update_traces(marker=dict(line=dict(width=0)))
                st.plotly_chart(fig_zone_pos, use_container_width=True)

        # Pitch map — scatter of run start positions
        st.markdown("#### Run start positions — pitch map")
        sample = runs_df.sample(min(3000, len(runs_df)), random_state=42)

        fig_pitch = go.Figure()

        # Pitch outline
        for shape in [
            dict(type="rect", x0=-52.5, x1=52.5, y0=-34, y1=34,
                 line=dict(color="#2a3f5f", width=2)),
            dict(type="line", x0=0, x1=0, y0=-34, y1=34,
                 line=dict(color="#2a3f5f", width=1)),
            dict(type="circle", x0=-9.15, x1=9.15, y0=-9.15, y1=9.15,
                 line=dict(color="#2a3f5f", width=1)),
            dict(type="rect", x0=-52.5, x1=-36, y0=-20.16, y1=20.16,
                 line=dict(color="#2a3f5f", width=1)),
            dict(type="rect", x0=36, x1=52.5, y0=-20.16, y1=20.16,
                 line=dict(color="#2a3f5f", width=1)),
        ]:
            fig_pitch.add_shape(**shape)

        fig_pitch.add_trace(go.Scatter(
            x=sample["start_x"],
            y=sample["start_y"],
            mode="markers",
            marker=dict(
                size=4,
                color=sample["peak_speed_kmh"],
                colorscale=[[0, "#1e3a5f"], [0.5, "#2196f3"], [1.0, "#4fc3f7"]],
                colorbar=dict(title="Peak speed (km/h)", thickness=10),
                opacity=0.6,
                line=dict(width=0),
            ),
            hovertemplate="x: %{x:.1f}m<br>y: %{y:.1f}m<extra></extra>",
        ))

        fig_pitch.update_layout(
            plot_bgcolor="#0a1628",
            paper_bgcolor=PAPER_BG,
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(range=[-55, 55], showgrid=False, zeroline=False,
                       title="metres", tickfont=dict(color=TEXT_COLOR, size=9)),
            yaxis=dict(range=[-37, 37], showgrid=False, zeroline=False,
                       scaleanchor="x", scaleratio=1,
                       tickfont=dict(color=TEXT_COLOR, size=9)),
        )
        st.plotly_chart(fig_pitch, use_container_width=True)
        st.caption("Sample of 3,000 HSR run start positions. Colour = peak speed.")

    else:
        st.info("Pitch zone analysis requires start_x/start_y columns in hsr_runs.csv")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center;font-family:\'DM Mono\',monospace;font-size:11px;'
    'color:#2d3748;">GradientSports FIFA World Cup 2022 · fast-forward-football · '
    'Databricks Delta · Built for USSF Data Scientist portfolio</p>',
    unsafe_allow_html=True
)
