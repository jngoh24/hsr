"""
Definition Comparison: Relative HSR vs Industry Standard (20 km/h)
===================================================================
Quantifies exactly what changes when you switch from the absolute
20 km/h threshold to the 75%-of-vmax relative definition.

This module is designed to produce the comparison table you'd show
in an interview or in the app's "methodology" section.
"""

import polars as pl
from hsr_metric import (
    compute_speed,
    compute_vmax,
    tag_relative_hsr_frames,
    extract_hsr_runs,
    DEFAULT_THRESHOLD_PCT,
    MIN_DURATION_SEC,
    FRAME_RATE_HZ,
)

INDUSTRY_THRESHOLD_KMH: float = 20.0


def extract_absolute_hsr_runs(
    tracking_df: pl.DataFrame,
    threshold_kmh: float = INDUSTRY_THRESHOLD_KMH,
    min_duration_sec: float = MIN_DURATION_SEC,
    frame_rate_hz: float = FRAME_RATE_HZ,
) -> pl.DataFrame:
    """
    Extract HSR runs using the industry-standard absolute threshold.

    This mirrors extract_hsr_runs() but uses a fixed km/h cutoff
    instead of the relative personal maximum approach.

    Parameters
    ----------
    tracking_df : pl.DataFrame
        Must have speed_kmh column. Output of compute_speed().
    threshold_kmh : float
        Absolute speed threshold in km/h. Default 20.0 (industry standard).
    min_duration_sec : float
        Minimum continuous duration to count as a run.
    frame_rate_hz : float
        Frame rate of the tracking data.

    Returns
    -------
    pl.DataFrame
        Same schema as extract_hsr_runs() output, with pct_of_vmax = null
        (not applicable for the absolute definition).
    """
    min_frames = int(min_duration_sec * frame_rate_hz)

    hsr_only = tracking_df.filter(
        pl.col("speed_kmh").is_not_null()
        & (pl.col("speed_kmh") >= threshold_kmh)
        & (pl.col("player_id") != "ball")
    )

    if hsr_only.is_empty():
        return pl.DataFrame(schema={
            "player_id": pl.Utf8, "game_id": pl.Utf8, "period_id": pl.Int64,
            "run_id": pl.UInt32, "start_frame_id": pl.UInt32,
            "end_frame_id": pl.UInt32, "duration_sec": pl.Float32,
            "peak_speed_kmh": pl.Float32, "mean_speed_kmh": pl.Float32,
            "pct_of_vmax": pl.Float32, "distance_m": pl.Float32,
            "vmax_kmh": pl.Float32,
            "start_x": pl.Float32, "start_y": pl.Float32,
            "end_x": pl.Float32, "end_y": pl.Float32,
        })

    runs = (
        hsr_only
        .sort(["player_id", "game_id", "frame_id"])
        .with_columns([
            pl.col("frame_id").shift(1).over(["player_id", "game_id"]).alias("prev_frame_id"),
        ])
        .with_columns([
            (
                pl.col("prev_frame_id").is_null()
                | (pl.col("frame_id") - pl.col("prev_frame_id") > 1)
            )
            .cast(pl.UInt32)
            .alias("run_boundary")
        ])
        .with_columns([
            pl.col("run_boundary")
              .cum_sum()
              .over(["player_id", "game_id"])
              .alias("run_id")
        ])
        .group_by(["player_id", "game_id", "period_id", "run_id"])
        .agg([
            pl.col("frame_id").min().alias("start_frame_id"),
            pl.col("frame_id").max().alias("end_frame_id"),
            pl.col("frame_id").count().alias("n_frames"),
            pl.col("speed_kmh").max().alias("peak_speed_kmh"),
            pl.col("speed_kmh").mean().alias("mean_speed_kmh"),
            pl.col("x").first().alias("start_x"),
            pl.col("y").first().alias("start_y"),
            pl.col("x").last().alias("end_x"),
            pl.col("y").last().alias("end_y"),
        ])
        .filter(pl.col("n_frames") >= min_frames)
        .with_columns([
            (pl.col("n_frames") / frame_rate_hz).cast(pl.Float32).alias("duration_sec"),
            pl.lit(None).cast(pl.Float32).alias("pct_of_vmax"),
            (pl.col("mean_speed_kmh") * pl.col("n_frames") / frame_rate_hz / 3.6)
            .cast(pl.Float32).alias("distance_m"),
            pl.lit(None).cast(pl.Float32).alias("vmax_kmh"),
        ])
        .drop(["n_frames"])
        .sort(["player_id", "game_id", "start_frame_id"])
    )

    return runs


def compare_definitions(
    tracking_df: pl.DataFrame,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    absolute_threshold_kmh: float = INDUSTRY_THRESHOLD_KMH,
) -> pl.DataFrame:
    """
    Side-by-side comparison of relative vs absolute HSR definition per player.

    Returns a single DataFrame showing what changes — who gains runs, who
    loses them, and the players most affected by the definition change.

    Parameters
    ----------
    tracking_df : pl.DataFrame
        Raw fastforward tracking with speed_kmh already computed,
        or raw tracking (speed will be computed internally).
    threshold_pct : float
        Relative threshold fraction (default 0.75).
    absolute_threshold_kmh : float
        Absolute threshold in km/h (default 20.0).

    Returns
    -------
    pl.DataFrame
        One row per player. Key columns:
            player_id
            vmax_kmh              : personal top speed
            relative_threshold_kmh: absolute speed the relative threshold resolves to
            runs_relative         : HSR count under relative definition
            runs_absolute         : HSR count under absolute definition
            run_delta             : runs_relative - runs_absolute
            pct_change            : percentage change in HSR count
            category              : "gained", "lost", "unchanged"

        Sorted by absolute value of run_delta descending — most impacted
        players first.
    """
    # Ensure speed column exists
    if "speed_kmh" not in tracking_df.columns:
        tracking_df = compute_speed(tracking_df)

    # Relative definition
    vmax_df = compute_vmax(tracking_df)
    tagged = tag_relative_hsr_frames(tracking_df, vmax_df, threshold_pct)
    relative_runs = extract_hsr_runs(tagged)

    relative_counts = (
        relative_runs
        .group_by("player_id")
        .agg(pl.len().alias("runs_relative"))
    )

    # Absolute definition
    absolute_runs = extract_absolute_hsr_runs(tracking_df, absolute_threshold_kmh)

    absolute_counts = (
        absolute_runs
        .group_by("player_id")
        .agg(pl.len().alias("runs_absolute"))
    )

    # Join both on vmax for context
    comparison = (
        vmax_df.select(["player_id", "vmax_kmh", "low_confidence"])
        .join(relative_counts, on="player_id", how="left")
        .join(absolute_counts, on="player_id", how="left")
        .with_columns([
            pl.col("runs_relative").fill_null(0),
            pl.col("runs_absolute").fill_null(0),
            (pl.col("vmax_kmh") * threshold_pct)
            .cast(pl.Float32)
            .alias("relative_threshold_kmh"),
        ])
        .with_columns([
            (pl.col("runs_relative") - pl.col("runs_absolute")).alias("run_delta"),
            pl.when(pl.col("runs_absolute") > 0)
            .then(
                ((pl.col("runs_relative") - pl.col("runs_absolute"))
                 / pl.col("runs_absolute") * 100)
                .round(1)
            )
            .otherwise(None)
            .alias("pct_change"),
        ])
        .with_columns([
            pl.when(pl.col("run_delta") > 0).then(pl.lit("gained"))
            .when(pl.col("run_delta") < 0).then(pl.lit("lost"))
            .otherwise(pl.lit("unchanged"))
            .alias("category"),
        ])
        .filter(pl.col("low_confidence").not_())
        .sort(pl.col("run_delta").abs(), descending=True)
    )

    return comparison
