"""
Relative High-Speed Run (HSR) Metric
=====================================
Redefines high-speed running as effort relative to each player's
personal maximum speed, rather than an absolute population threshold.

Definition
----------
A high-speed run is any continuous period where a player maintains
>= 75% of their personal recorded v_max for at least 1 second.

Compared to the industry standard (flat 20 km/h cutoff), this definition:
  - Accounts for individual speed profiles across positions and athletes
  - Is physiologically grounded (analogous to %HRmax, %VO2max)
  - Produces fairer cross-player comparisons at a national team level
  - Surfaces high-effort runs from lower-speed players missed by flat thresholds

Data source
-----------
GradientSports (formerly PFF) FIFA World Cup 2022 open tracking data,
loaded via fast-forward-football:
    pip install fast-forward-football

    from fastforward import gradientsports
    dataset = gradientsports.load_tracking(
        raw_data="{game_id}.jsonl.bz2",
        meta_data="{game_id}_metadata.json",
        roster_data="{game_id}_rosters.json",
    )

Schema expected (fastforward long layout)
-----------------------------------------
Columns: game_id, frame_id, period_id, timestamp, ball_state,
         ball_owning_team_id, team_id, player_id, x, y, z

Speed is derived from positional displacement between consecutive frames.
"""

import polars as pl
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_RATE_HZ: float = 25.0          # GradientSports broadcast tracking
KMH_PER_MS: float = 3.6              # m/s -> km/h conversion
DEFAULT_THRESHOLD_PCT: float = 0.75  # 75% of personal v_max
MIN_DURATION_SEC: float = 1.0        # minimum run duration in seconds
MIN_FRAMES_FOR_VMAX: int = 250       # ~10 seconds minimum exposure to trust v_max


# ---------------------------------------------------------------------------
# Step 1: Derive speed from positional displacement
# ---------------------------------------------------------------------------

def compute_speed(tracking_df: pl.DataFrame) -> pl.DataFrame:
    """
    Derive per-frame speed in km/h from x/y displacement.

    GradientSports tracking data contains x, y coordinates in metres
    (CDF format, origin at centre). Speed is not provided directly —
    we compute it as Euclidean displacement between consecutive frames,
    divided by the frame interval, converted to km/h.

    Parameters
    ----------
    tracking_df : pl.DataFrame
        Long-format tracking DataFrame from fastforward.
        Must contain: player_id, frame_id, x, y.

    Returns
    -------
    pl.DataFrame
        Same rows with added column `speed_kmh` (Float32).
        First frame per player will have speed_kmh = null (no prior frame).

    Notes
    -----
    - Ball rows (player_id == "ball") are retained but speed is computed
      the same way — filter them downstream if not needed.
    - Frame gaps > 1 (dropped frames) will produce inflated speed values.
      We clip at 45 km/h as a sanity ceiling (beyond human capability).
    """
    frame_interval_sec = 1.0 / FRAME_RATE_HZ

    return (
        tracking_df
        .sort(["player_id", "frame_id"])
        .with_columns([
            pl.col("x").shift(1).over("player_id").alias("x_prev"),
            pl.col("y").shift(1).over("player_id").alias("y_prev"),
            pl.col("frame_id").shift(1).over("player_id").alias("frame_id_prev"),
        ])
        .with_columns([
            # Only compute speed where frames are consecutive (no gaps)
            pl.when(
                (pl.col("frame_id") - pl.col("frame_id_prev") == 1)
                & pl.col("x_prev").is_not_null()
            )
            .then(
                (
                    ((pl.col("x") - pl.col("x_prev")) ** 2
                     + (pl.col("y") - pl.col("y_prev")) ** 2)
                    .sqrt()
                    / frame_interval_sec
                    * KMH_PER_MS
                ).clip(0.0, 45.0)  # sanity ceiling
            )
            .otherwise(None)
            .cast(pl.Float32)
            .alias("speed_kmh")
        ])
        .drop(["x_prev", "y_prev", "frame_id_prev"])
    )


# ---------------------------------------------------------------------------
# Step 2: Compute personal v_max per player (per tournament or per game)
# ---------------------------------------------------------------------------

def compute_vmax(
    tracking_df: pl.DataFrame,
    percentile: float = 99.5,
    min_frames: int = MIN_FRAMES_FOR_VMAX,
    group_by_game: bool = False,
) -> pl.DataFrame:
    """
    Compute each player's personal maximum speed (v_max).

    Rather than taking the single peak frame (which may be a tracking
    artefact or noise spike), we use a high percentile of observed speeds.
    99.5th percentile is robust to noise while capturing genuine top-end
    effort.

    Parameters
    ----------
    tracking_df : pl.DataFrame
        Must already have `speed_kmh` column (output of compute_speed).
        Must contain `player_id` and optionally `game_id`.
    percentile : float, default 99.5
        Percentile of non-null speed values to use as v_max.
        99.5 is robust to noise while still capturing genuine top-end effort.
    min_frames : int, default 250
        Minimum number of non-null speed frames required to trust a v_max
        estimate. Players with fewer frames are flagged as low_confidence.
        250 frames ≈ 10 seconds of tracking at 25 Hz.
    group_by_game : bool, default False
        If True, compute v_max per (player_id, game_id) separately.
        If False (default), pool all games for a tournament-level v_max —
        recommended for World Cup data where each player appears in ~7 games.

    Returns
    -------
    pl.DataFrame
        One row per player (or per player+game if group_by_game=True).
        Columns: player_id, [game_id], vmax_kmh, frame_count, low_confidence.

    Notes
    -----
    The `low_confidence` flag is important for broadcast tracking (GradientSports
    uses computer vision on TV footage, not full-pitch optical sensors). Players
    who appear infrequently — substitutes, injured players, those off-camera —
    may have unreliable v_max estimates. Flag these and exclude from league tables.
    """
    group_cols = ["player_id", "game_id"] if group_by_game else ["player_id"]

    speed_only = tracking_df.filter(
        pl.col("speed_kmh").is_not_null()
        & (pl.col("player_id") != "ball")
    )

    return (
        speed_only
        .group_by(group_cols)
        .agg([
            pl.col("speed_kmh")
              .quantile(percentile / 100.0, interpolation="linear")
              .alias("vmax_kmh"),
            pl.col("speed_kmh").count().alias("frame_count"),
        ])
        .with_columns([
            (pl.col("frame_count") < min_frames).alias("low_confidence"),
        ])
        .sort(group_cols)
    )


# ---------------------------------------------------------------------------
# Step 3: Tag each frame as in / out of a relative HSR
# ---------------------------------------------------------------------------

def tag_relative_hsr_frames(
    tracking_df: pl.DataFrame,
    vmax_df: pl.DataFrame,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
) -> pl.DataFrame:
    """
    Join v_max back onto tracking data and flag frames above the
    relative speed threshold.

    Parameters
    ----------
    tracking_df : pl.DataFrame
        Must have speed_kmh column.
    vmax_df : pl.DataFrame
        Output of compute_vmax(). Must have player_id and vmax_kmh.
        If vmax_df has game_id column, join is per (player_id, game_id).
    threshold_pct : float, default 0.75
        Fraction of v_max required to qualify as a high-speed frame.

    Returns
    -------
    pl.DataFrame
        Original tracking_df with added columns:
            vmax_kmh       : player's personal max speed
            speed_threshold: absolute speed required (vmax * threshold_pct)
            is_hsr_frame   : bool, True when speed >= threshold and sustained
    """
    join_cols = ["player_id", "game_id"] if "game_id" in vmax_df.columns else ["player_id"]

    return (
        tracking_df
        .join(
            vmax_df.select(join_cols + ["vmax_kmh", "low_confidence"]),
            on=join_cols,
            how="left",
        )
        .with_columns([
            (pl.col("vmax_kmh") * threshold_pct)
            .cast(pl.Float32)
            .alias("speed_threshold"),
        ])
        .with_columns([
            (
                pl.col("speed_kmh").is_not_null()
                & (pl.col("speed_kmh") >= pl.col("speed_threshold"))
                & pl.col("low_confidence").not_()
            )
            .alias("is_hsr_frame")
        ])
    )


# ---------------------------------------------------------------------------
# Step 4: Segment continuous HSR frames into discrete runs
# ---------------------------------------------------------------------------

def extract_hsr_runs(
    tagged_df: pl.DataFrame,
    min_duration_sec: float = MIN_DURATION_SEC,
    frame_rate_hz: float = FRAME_RATE_HZ,
) -> pl.DataFrame:
    """
    Group consecutive is_hsr_frame=True frames into discrete HSR events
    and filter to those meeting minimum duration.

    This is where the "maintained for at least 1 second" part of the
    definition is enforced. A run is broken if:
      - is_hsr_frame becomes False (player drops below threshold)
      - frame_id is not consecutive (tracking gap / out of view)

    Parameters
    ----------
    tagged_df : pl.DataFrame
        Output of tag_relative_hsr_frames(). Must have:
        player_id, game_id, period_id, frame_id, timestamp,
        speed_kmh, speed_threshold, vmax_kmh, is_hsr_frame, x, y.
    min_duration_sec : float, default 1.0
        Minimum run duration in seconds to count as a valid HSR.
    frame_rate_hz : float, default 25.0
        Frames per second — used to convert frame count to duration.

    Returns
    -------
    pl.DataFrame
        One row per valid HSR event. Columns:
            player_id, game_id, period_id,
            run_id           : unique identifier per run per player per game
            start_frame_id   : first frame of the run
            end_frame_id     : last frame of the run
            duration_sec     : run duration in seconds
            peak_speed_kmh   : maximum speed observed during the run
            mean_speed_kmh   : mean speed during the run
            vmax_kmh         : player's personal max speed
            pct_of_vmax      : peak_speed / vmax (how hard was this run)
            distance_m       : approximate distance covered (speed × duration)
            start_x, start_y : pitch coordinates at run start
            end_x, end_y     : pitch coordinates at run end
    """
    min_frames = int(min_duration_sec * frame_rate_hz)

    # Detect run boundaries: new run starts when is_hsr_frame transitions
    # from False→True, or when there's a frame gap (non-consecutive frame_ids)
    hsr_only = tagged_df.filter(pl.col("is_hsr_frame"))

    if hsr_only.is_empty():
        return pl.DataFrame(schema={
            "player_id": pl.Utf8, "game_id": pl.Utf8, "period_id": pl.Int32,
            "run_id": pl.UInt32, "start_frame_id": pl.UInt32,
            "end_frame_id": pl.UInt32, "duration_sec": pl.Float32,
            "peak_speed_kmh": pl.Float32, "mean_speed_kmh": pl.Float32,
            "vmax_kmh": pl.Float32, "pct_of_vmax": pl.Float32,
            "distance_m": pl.Float32,
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
            # New run starts when frames are non-consecutive or no prior frame
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
            pl.col("vmax_kmh").first().alias("vmax_kmh"),
            pl.col("x").first().alias("start_x"),
            pl.col("y").first().alias("start_y"),
            pl.col("x").last().alias("end_x"),
            pl.col("y").last().alias("end_y"),
        ])
        .filter(pl.col("n_frames") >= min_frames)
        .with_columns([
            (pl.col("n_frames") / frame_rate_hz).cast(pl.Float32).alias("duration_sec"),
            (pl.col("peak_speed_kmh") / pl.col("vmax_kmh")).cast(pl.Float32).alias("pct_of_vmax"),
            # distance ≈ mean_speed (km/h) × duration (sec) / 3.6
            (pl.col("mean_speed_kmh") * pl.col("n_frames") / frame_rate_hz / KMH_PER_MS)
            .cast(pl.Float32)
            .alias("distance_m"),
        ])
        .drop(["n_frames"])
        .sort(["player_id", "game_id", "start_frame_id"])
    )

    return runs


# ---------------------------------------------------------------------------
# Step 5: Aggregate to player-level summary
# ---------------------------------------------------------------------------

def summarise_hsr_per_player(
    runs_df: pl.DataFrame,
    vmax_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Aggregate HSR runs to player-level summary statistics.

    This is the output table you'd share with coaching or performance staff:
    who is making the most high-speed efforts relative to their own ceiling?

    Parameters
    ----------
    runs_df : pl.DataFrame
        Output of extract_hsr_runs().
    vmax_df : pl.DataFrame
        Output of compute_vmax() — joined to add v_max and confidence flag.

    Returns
    -------
    pl.DataFrame
        One row per player. Columns:
            player_id
            vmax_kmh            : personal top speed
            low_confidence      : whether v_max estimate is reliable
            total_runs          : total HSR count across all games
            total_distance_m    : total distance covered in HSR
            mean_duration_sec   : average run duration
            mean_pct_of_vmax    : average intensity of runs (how close to ceiling)
            mean_peak_speed_kmh : average peak speed per run
            runs_per_game       : normalised by games appeared in
    """
    games_per_player = (
        runs_df
        .group_by("player_id")
        .agg(pl.col("game_id").n_unique().alias("n_games"))
    )

    summary = (
        runs_df
        .group_by("player_id")
        .agg([
            pl.len().alias("total_runs"),
            pl.col("distance_m").sum().alias("total_distance_m"),
            pl.col("duration_sec").mean().alias("mean_duration_sec"),
            pl.col("pct_of_vmax").mean().alias("mean_pct_of_vmax"),
            pl.col("peak_speed_kmh").mean().alias("mean_peak_speed_kmh"),
        ])
        .join(games_per_player, on="player_id", how="left")
        .join(
            vmax_df.select(["player_id", "vmax_kmh", "low_confidence"]),
            on="player_id",
            how="left",
        )
        .with_columns([
            (pl.col("total_runs") / pl.col("n_games"))
            .cast(pl.Float32)
            .alias("runs_per_game"),
        ])
        .sort("total_runs", descending=True)
    )

    return summary


# ---------------------------------------------------------------------------
# Step 6: Full pipeline — single entry point
# ---------------------------------------------------------------------------

def run_relative_hsr_pipeline(
    tracking_df: pl.DataFrame,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    vmax_percentile: float = 99.5,
    min_duration_sec: float = MIN_DURATION_SEC,
    group_vmax_by_game: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    End-to-end pipeline: raw tracking -> v_max -> tagged frames -> runs -> summary.

    Parameters
    ----------
    tracking_df : pl.DataFrame
        Raw long-format fastforward TrackingDataset.tracking DataFrame.
    threshold_pct : float
        Fraction of v_max to use as high-speed threshold (default 0.75).
    vmax_percentile : float
        Percentile to use for v_max estimation (default 99.5).
    min_duration_sec : float
        Minimum run duration in seconds (default 1.0).
    group_vmax_by_game : bool
        If True, compute v_max per game rather than across tournament.

    Returns
    -------
    tuple of (vmax_df, runs_df, summary_df)
        vmax_df    : player-level v_max estimates
        runs_df    : individual HSR event records
        summary_df : aggregated player summary
    """
    print(f"[1/5] Computing speed from positional displacement...")
    with_speed = compute_speed(tracking_df)

    print(f"[2/5] Estimating personal v_max (p{vmax_percentile}) per player...")
    vmax_df = compute_vmax(with_speed, percentile=vmax_percentile, group_by_game=group_vmax_by_game)
    print(f"      Found v_max for {vmax_df.height} players "
          f"({vmax_df.filter(pl.col('low_confidence')).height} flagged low-confidence)")

    print(f"[3/5] Tagging HSR frames (threshold: {threshold_pct*100:.0f}% of v_max)...")
    tagged = tag_relative_hsr_frames(with_speed, vmax_df, threshold_pct)

    print(f"[4/5] Extracting runs (min duration: {min_duration_sec}s)...")
    runs_df = extract_hsr_runs(tagged, min_duration_sec)
    print(f"      Extracted {runs_df.height} valid HSR events")

    print(f"[5/5] Aggregating to player summary...")
    summary_df = summarise_hsr_per_player(runs_df, vmax_df)

    return vmax_df, runs_df, summary_df
