from pathlib import Path
from typing import List
import time

import pandas as pd


class RollingStdevPipeline:
    """Pipeline to compute rolling standard deviations."""

    PRICE_COLUMNS: List[str] = ["bid", "mid", "ask"]
    STDEV_COLUMNS: List[str] = ["bid_stdev", "mid_stdev", "ask_stdev"]
    START_TIME = pd.Timestamp("2021-11-20 00:00:00")
    END_TIME = pd.Timestamp("2021-11-23 09:00:00")

    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir

    def load_data(self) -> pd.DataFrame:
        """Load input dataset."""
        return pd.read_parquet(
            self.input_dir / "stdev_price_data.parq.gzip"
        )

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Sort data and detect contiguity breaks."""
        df = df.sort_values(
            ["security_id", "snap_time"]
        ).reset_index(drop=True)

        df["snap_time"] = pd.to_datetime(df["snap_time"])

        time_diffs = df.groupby("security_id")["snap_time"].diff()

        df["contiguity_broken"] = (
            (time_diffs != pd.Timedelta("1h")) & time_diffs.notna()
        )

        df["sequence_id"] = (
            df.groupby("security_id")["contiguity_broken"].cumsum()
        )

        return df

    @staticmethod
    def compute_stdev(df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling standard deviations."""
        for col in RollingStdevPipeline.PRICE_COLUMNS:
            df[f"{col}_stdev"] = (
                df.groupby(["security_id", "sequence_id"])[col]
                .rolling(window=20, min_periods=20)
                .std()
                .reset_index(level=[0, 1], drop=True)
            )

        for col in RollingStdevPipeline.STDEV_COLUMNS:
            df[col] = (
                df
                .groupby(["security_id"])[col]
                .ffill()
            )

        output_cols = ["security_id", "snap_time", *RollingStdevPipeline.STDEV_COLUMNS]

        # Generate the hourly intervals for each security id

        # Unique security ids
        security_ids = df["security_id"].unique()

        # Hourly intervals from start time to end time
        hours = pd.date_range(RollingStdevPipeline.START_TIME, RollingStdevPipeline.END_TIME, freq="h")

        hourly_intervals = (
            pd.MultiIndex.from_product(
                [security_ids,hours],
                names=["security_id","snap_time"],
            )
            .to_frame(index=False)
        )

        # Merge values from initial df (real + forward filled stdevs)
        hourly_intervals = pd.merge(
            hourly_intervals,
            df[output_cols],
            on=["security_id", "snap_time"],
            how="left",
        )

        # Add last known snap_time for each security id, as it is previously forward filled this will have the most recent stdev from contiguous value
        last_value = (
            df[df["snap_time"] < RollingStdevPipeline.START_TIME]
            .sort_values("snap_time")
            .groupby("security_id")
            .tail(1)
        )[output_cols]

        output = pd.concat(
            [
                last_value,
                hourly_intervals,
            ],
            ignore_index=True,
        ).sort_values(["security_id", "snap_time"])

        # Forward fill
        output[RollingStdevPipeline.STDEV_COLUMNS] = (
            output.groupby("security_id")[RollingStdevPipeline.STDEV_COLUMNS].ffill()
        )

        # Filter out the last known snap time
        output = output[output["snap_time"] >= RollingStdevPipeline.START_TIME]

        return output

    @staticmethod
    def validate_outputs(df: pd.DataFrame) -> None:
        """Validate rolling stdev outputs."""

        # No duplicate snap times per security
        duplicated = df.duplicated(
            subset=["security_id", "snap_time"]
        )
        if duplicated.any():
            rows = df.loc[duplicated].index.tolist()
            raise ValueError(
                f"Duplicate (security_id, snap_time) rows detected: {rows}"
            )

        # snap_time strictly increasing per security
        time_diff = df.groupby("security_id")["snap_time"].diff()
        if (time_diff <= pd.Timedelta(0)).any():
            rows = df.loc[
                time_diff <= pd.Timedelta(0)
            ].index.tolist()
            raise ValueError(
                "snap_time is not strictly increasing within security_id "
                f"(rows: {rows})"
            )

        # stdev only valid when 20 contiguous points exist
        for col in ["bid", "mid", "ask"]:
            stdev_col = f"{col}_stdev"

            invalid = (
                df[stdev_col].notna()
                & (df.groupby("security_id")[stdev_col]
                .transform("count") < 20)
            )

            if invalid.any():
                rows = df.loc[invalid].index.tolist()
                raise ValueError(
                    f"{stdev_col} populated with <20 contiguous observations "
                    f"(rows: {rows})"
                )

        # Non-negative stdev
        for col in ["bid_stdev", "mid_stdev", "ask_stdev"]:
            if (df[col] < 0).any():
                rows = df.loc[df[col] < 0].index.tolist()
                raise ValueError(
                    f"Negative values found in {col} (rows: {rows})"
                )
        
    def run(self) -> None:
        """Execute the full pipeline."""
        start = time.perf_counter()

        df = self.load_data()
        df = self.preprocess(df)
        df = self.compute_stdev(df)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "rolling_stdev_prices.csv"
        pipeline_elapsed = time.perf_counter() - start
        print(f"Pipeline completed in {pipeline_elapsed:.3f}s")
        df.to_csv(output_path, index=False)

        final_elapsed = time.perf_counter() - pipeline_elapsed
        print(f"Writing completed in {final_elapsed:.3f}s")
        print(f"Output written to {output_path}")

        self.validate_outputs(df)

def main() -> None:
    """Entry point (uses relative paths)."""
    input_dir = Path("data")
    output_dir = Path("results")

    pipeline = RollingStdevPipeline(input_dir, output_dir)
    pipeline.run()


if __name__ == "__main__":
    main()