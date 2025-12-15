from pathlib import Path
from typing import Tuple
import time

import pandas as pd

class RatesPipeline:
    """Pipeline to compute converted FX prices."""

    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir

    def load_data(self):
        """Load input datasets."""
        rates_ccy = pd.read_csv(self.input_dir / "rates_ccy_data.csv")
        rates_spot = pd.read_parquet(self.input_dir / "rates_spot_rate_data.parq.gzip")
        rates_price = pd.read_parquet(self.input_dir / "rates_price_data.parq.gzip")

        return rates_price, rates_ccy, rates_spot

    @staticmethod
    def preprocess(
        rates_price: pd.DataFrame,
        rates_ccy: pd.DataFrame,
        rates_spot: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare datasets for joining."""
        rates = pd.merge(
            rates_price,
            rates_ccy,
            on="ccy_pair",
            how="left",
        )

        for df in (rates, rates_spot):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        rates = rates.sort_values("timestamp")
        rates_spot = rates_spot.sort_values("timestamp")

        return rates, rates_spot

    @staticmethod
    def compute_new_price(
        rates: pd.DataFrame,
        rates_spot: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute converted prices using preceding spot rates."""
        merged = pd.merge_asof(
            rates,
            rates_spot,
            on="timestamp",
            by="ccy_pair",
            direction="backward",
            tolerance=pd.Timedelta("1h"),
        )

        merged["new_price"] = (
            merged["price"] / merged["conversion_factor"]
            + merged["spot_mid_rate"]
        )

        merged.loc[
            merged["convert_price"] == False, "new_price"
        ] = merged.loc[
            merged["convert_price"] == False, "price"
        ]

        return merged

    @staticmethod
    def add_comments(df: pd.DataFrame) -> pd.DataFrame:
        """Annotate rows with data quality comments."""
        unsupported = df["convert_price"].isna()
        no_spot = (df["spot_mid_rate"].isna()) & (~unsupported) & (df["convert_price"])

        df.loc[unsupported, "comments"] = "ccy_pair not supported"
        df.loc[no_spot, "comments"] = "no preceding data for spot rates"
        df["comments"] = df["comments"].fillna("")

        return df

    @staticmethod
    def validate_inputs(
        rates_ccy: pd.DataFrame,
    ) -> None:
        """Validate reference input data."""

        # Unique ccy_pair check
        duplicated = rates_ccy["ccy_pair"].duplicated()
        if duplicated.any():
            dupes = rates_ccy.loc[duplicated, "ccy_pair"].unique()
            raise ValueError(
                f"Duplicate ccy_pair values in rates_ccy_data: {dupes}"
            )
        
        print("Input validated successfully.")

    @staticmethod
    def validate_outputs(df: pd.DataFrame) -> None:
        """Validate output new_price and comments."""

        # convert_price == False → new_price must equal price
        mask_no_conversion = df["convert_price"] == False

        if mask_no_conversion.any():
            mismatch = (
                df.loc[mask_no_conversion, "new_price"]
                != df.loc[mask_no_conversion, "price"]
            )

            if mismatch.any():
                rows = df.loc[mask_no_conversion & mismatch].index.tolist()
                raise ValueError(
                    "Validation failed: new_price != price when convert_price == False "
                    f"(rows: {rows})"
                )

        # Unsupported ccy_pair
        # convert_price == NaN → new_price == NaN and comment == 'ccy_pair not supported'
        mask_unsupported = df["convert_price"].isna()

        if mask_unsupported.any():
            invalid_price = df.loc[mask_unsupported, "new_price"].notna()
            invalid_comment = (
                df.loc[mask_unsupported, "comments"]
                != "ccy_pair not supported"
            )

            if invalid_price.any() or invalid_comment.any():
                rows = df.loc[
                    mask_unsupported & (invalid_price | invalid_comment)
                ].index.tolist()
                raise ValueError(
                    "Validation failed: unsupported ccy_pair rows must have "
                    "new_price == NaN and comment == 'ccy_pair not supported' "
                    f"(rows: {rows})"
                )

        # Missing spot_mid_rate
        # spot_mid_rate == NaN & convert_price == True →
        # new_price == NaN and correct comment
        mask_no_spot = (
            df["spot_mid_rate"].isna()
            & df["convert_price"]
        )

        if mask_no_spot.any():
            invalid_price = df.loc[mask_no_spot, "new_price"].notna()
            invalid_comment = (
                df.loc[mask_no_spot, "comments"]
                != "no preceding data for spot rates"
            )

            if invalid_price.any() or invalid_comment.any():
                rows = df.loc[
                    mask_no_spot & (invalid_price | invalid_comment)
                ].index.tolist()
                raise ValueError(
                    "Validation failed: rows without spot_mid_rate must have "
                    "new_price == NaN and comment == "
                    "'no preceding data for spot rates' "
                    f"(rows: {rows})"
                )
            
        print("Output validated successfully.")

    def run(self) -> None:
        """Execute the full pipeline."""
        start = time.perf_counter()

        rates_price, rates_ccy, rates_spot = self.load_data()

        self.validate_inputs(rates_ccy)

        rates, rates_spot = self.preprocess(
            rates_price, rates_ccy, rates_spot
        )

        result = self.compute_new_price(rates, rates_spot)

        result = self.add_comments(result)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "rates_with_new_price.csv"
        
        pipeline_elapsed = time.perf_counter() - start
        print(f"Pipeline completed in {pipeline_elapsed:.3f}s")
        result.to_csv(output_path, index=False)

        final_elapsed = time.perf_counter() - pipeline_elapsed
        print(f"Writing completed in {final_elapsed:.3f}s")
        print(f"Output written to {output_path}")

        self.validate_outputs(result)

def main() -> None:
    """Execute pipeline."""
    input_dir = Path("data")
    output_dir = Path("results")

    pipeline = RatesPipeline(input_dir, output_dir)
    pipeline.run()


if __name__ == "__main__":

    main()
