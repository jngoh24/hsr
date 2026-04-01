# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver — Feature Engineering
# MAGIC
# MAGIC Reads Bronze Delta tables and produces Silver Delta tables enriched with:
# MAGIC - Per-frame speed (km/h) derived from x/y displacement at 25 Hz
# MAGIC - Personal v_max per player (tournament-level p99.5 across all 64 games)
# MAGIC - Relative HSR threshold per player (75% of personal v_max)
# MAGIC - Frame-level HSR flag (`is_hsr_frame`)
# MAGIC - Low-confidence flag for players with insufficient tracking exposure
# MAGIC
# MAGIC **Medallion layer**: Bronze → **Silver** → Gold
# MAGIC
# MAGIC **Inputs**
# MAGIC ```
# MAGIC delta/tracking_bronze/   ← written by Notebook 01
# MAGIC ```
# MAGIC
# MAGIC **Outputs**
# MAGIC ```
# MAGIC delta/tracking_silver/   ← 82M rows + speed + HSR tags
# MAGIC delta/vmax_per_player/   ← one row per player, v_max + threshold
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## 1 · Configuration

# COMMAND ----------

# ---------------------------------------------------------------------------
# Delta paths — must match Notebook 01
# ---------------------------------------------------------------------------
DELTA_BASE            = "dbfs:/mnt/cinqai_outputs/jn_adhoc/hsr/delta"
DELTA_TRACKING_BRONZE = f"{DELTA_BASE}/tracking_bronze"
DELTA_TRACKING_SILVER = f"{DELTA_BASE}/tracking_silver"
DELTA_VMAX            = f"{DELTA_BASE}/vmax_per_player"

# Source module path
SRC_PATH = "/dbfs/mnt/cinqai_outputs/jn_adhoc/hsr/src"

# ---------------------------------------------------------------------------
# Metric parameters
# ---------------------------------------------------------------------------
VMAX_PERCENTILE    = 99.5   # percentile used to estimate personal v_max
THRESHOLD_PCT      = 0.80   # 80% of v_max = relative HSR threshold
MIN_FRAMES_VMAX    = 250    # minimum frames to trust a v_max (~10 seconds)
FRAME_RATE_HZ      = 25.0   # GradientSports tracking rate

print(f"HSR definition : >= {THRESHOLD_PCT*100:.0f}% of personal v_max, sustained >= 1s")
print(f"v_max method   : p{VMAX_PERCENTILE} of observed speeds, min {MIN_FRAMES_VMAX} frames")
print(f"Frame rate     : {FRAME_RATE_HZ} Hz")

# COMMAND ----------

# MAGIC %md ## 2 · Load Bronze tracking table

# COMMAND ----------

from pyspark.sql import functions as F

bronze_df = (
    spark.read.format("delta").load(DELTA_TRACKING_BRONZE)
    .filter(F.col("player_id") != "ball")
)

n_rows    = bronze_df.count()
n_games   = bronze_df.select("game_id").distinct().count()
n_players = bronze_df.select("player_id").distinct().count()

print(f"Bronze: {n_rows:,} rows | {n_games} games | {n_players} players")

# COMMAND ----------

# MAGIC %md ## 3 · Compute per-frame speed via Pandas UDF
# MAGIC
# MAGIC Speed is derived from Euclidean x/y displacement between consecutive frames.
# MAGIC We use `applyInPandas` to run our tested Polars `compute_speed()` function
# MAGIC on each (player_id, game_id) partition — Polars for the math, Spark for
# MAGIC the distribution.

# COMMAND ----------

# Distribute source files to all Spark workers
spark.sparkContext.addPyFile(
    "dbfs:/mnt/cinqai_outputs/jn_adhoc/hsr/src/hsr_metric.py"
)
spark.sparkContext.addPyFile(
    "dbfs:/mnt/cinqai_outputs/jn_adhoc/hsr/src/hsr_comparison.py"
)

print("Source files distributed to workers")

# COMMAND ----------

# import sys
import pandas as pd
import polars as pl

# sys.path.insert(0, SRC_PATH)
from hsr_metric import compute_speed as _compute_speed

from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, LongType, FloatType
)

SPEED_SCHEMA = StructType([
    StructField("game_id",             StringType(),  True),
    StructField("frame_id",            LongType(),    True),
    StructField("period_id",           IntegerType(), True),
    StructField("timestamp",           LongType(),    True),
    StructField("ball_state",          StringType(),  True),
    StructField("ball_owning_team_id", StringType(),  True),
    StructField("team_id",             StringType(),  True),
    StructField("player_id",           StringType(),  True),
    StructField("x",                   FloatType(),   True),
    StructField("y",                   FloatType(),   True),
    StructField("z",                   FloatType(),   True),
    StructField("speed_kmh",           FloatType(),   True),
    StructField("_ingested_at",        StringType(),  True),
])

def add_speed_pandas(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Run Polars compute_speed() on one (player_id, game_id) partition.
    Converts Pandas → Polars → compute → Pandas.
    """
    pl_df  = pl.from_pandas(pdf)
    result = _compute_speed(pl_df)
    return result.to_pandas()

speed_df = (
    bronze_df
    .groupBy("player_id", "game_id")
    .applyInPandas(add_speed_pandas, schema=SPEED_SCHEMA)
)

print("Speed computation defined — will execute on write")

# COMMAND ----------

# MAGIC %md ## 4 · Compute tournament-level v_max per player
# MAGIC
# MAGIC v_max is pooled across all 64 games — more stable than per-game estimates,
# MAGIC especially for broadcast tracking where players may be off-camera in stretches.
# MAGIC
# MAGIC Uses p99.5 rather than raw maximum to guard against noise spikes.
# MAGIC Players with fewer than 250 tracked frames are flagged low_confidence.

# COMMAND ----------

# Materialise speed_df once so we don't recompute it twice
speed_df.cache()

vmax_df = (
    speed_df
    .filter(F.col("speed_kmh").isNotNull())
    .groupBy("player_id")
    .agg(
        F.percentile_approx(
            "speed_kmh",
            VMAX_PERCENTILE / 100.0,
            accuracy=10000
        ).alias("vmax_kmh"),
        F.count("speed_kmh").alias("frame_count"),
    )
    .withColumn(
        "low_confidence",
        F.col("frame_count") < MIN_FRAMES_VMAX
    )
    .withColumn(
        "speed_threshold_kmh",
        F.round(F.col("vmax_kmh") * THRESHOLD_PCT, 2)
    )
)

(
    vmax_df
    .write.format("delta")
    .mode("overwrite")
    .save(DELTA_VMAX)
)

print(f"v_max table written: {DELTA_VMAX}")
print(f"Players: {vmax_df.count()} total, "
      f"{vmax_df.filter(F.col('low_confidence')).count()} low confidence")

display(
    vmax_df
    .filter(~F.col("low_confidence"))
    .orderBy("vmax_kmh", ascending=False)
    .limit(20)
)

# COMMAND ----------

# MAGIC %md ## 5 · Tag HSR frames
# MAGIC
# MAGIC Join v_max back onto speed data and flag frames where the player
# MAGIC exceeds their personal relative threshold.

# COMMAND ----------

tagged_df = (
    speed_df
    .join(
        vmax_df.select("player_id", "vmax_kmh", "speed_threshold_kmh", "low_confidence"),
        on="player_id",
        how="left",
    )
    .withColumn(
        "is_hsr_frame",
        F.col("speed_kmh").isNotNull()
        & (F.col("speed_kmh") >= F.col("speed_threshold_kmh"))
        & (~F.col("low_confidence"))
    )
)

# COMMAND ----------

# MAGIC %md ## 6 · Write Silver Delta table

# COMMAND ----------

(
    tagged_df
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("game_id")
    .save(DELTA_TRACKING_SILVER)
)

# Unpersist cache
speed_df.unpersist()

total  = spark.read.format("delta").load(DELTA_TRACKING_SILVER).count()
hsr    = spark.read.format("delta").load(DELTA_TRACKING_SILVER).filter(F.col("is_hsr_frame")).count()

print(f"Silver written: {DELTA_TRACKING_SILVER}")
print(f"  Total frames : {total:,}")
print(f"  HSR frames   : {hsr:,}  ({hsr/total*100:.2f}%)")
print(f"\nExpected HSR %: 3-8% for elite players")
print(f"Ready for Notebook 03.")

# COMMAND ----------

# MAGIC %md ## 7 · Definition comparison preview
# MAGIC
# MAGIC Sanity check: relative definition vs industry standard 20 km/h flat threshold.

# COMMAND ----------

ABSOLUTE_THRESHOLD_KMH = 20.0

silver = spark.read.format("delta").load(DELTA_TRACKING_SILVER)
vmax   = spark.read.format("delta").load(DELTA_VMAX)

display(
    silver
    .filter(F.col("speed_kmh").isNotNull())
    .agg(
        F.count("*").alias("total_frames"),
        F.sum(
            (F.col("speed_kmh") >= ABSOLUTE_THRESHOLD_KMH).cast("int")
        ).alias("frames_absolute_def"),
        F.sum(
            F.col("is_hsr_frame").cast("int")
        ).alias("frames_relative_def"),
    )
    .withColumn("absolute_pct",
        F.round(F.col("frames_absolute_def") / F.col("total_frames") * 100, 2))
    .withColumn("relative_pct",
        F.round(F.col("frames_relative_def") / F.col("total_frames") * 100, 2))
    .withColumn("additional_frames_captured",
        F.col("frames_relative_def") - F.col("frames_absolute_def"))
)

# COMMAND ----------

