from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

RATES_SCRIPTS = PROJECT_ROOT / "rates_test" / "scripts"
STDEV_SCRIPTS = PROJECT_ROOT / "stdev_test" / "scripts"

sys.path.append(str(RATES_SCRIPTS))
sys.path.append(str(STDEV_SCRIPTS))

from rates import RatesPipeline
from stdev import RollingStdevPipeline

def main() -> None:

    print("Running rates pipeline")
    rates_pipeline = RatesPipeline(
        input_dir=PROJECT_ROOT / "rates_test" / "data",
        output_dir=PROJECT_ROOT / "rates_test" / "results",
    )
    rates_pipeline.run()
    print("Rates pipeline completed.")

    print("Running stdev pipeline")
    stdev_pipeline = RollingStdevPipeline(
        input_dir=PROJECT_ROOT / "stdev_test" / "data",
        output_dir=PROJECT_ROOT / "stdev_test" / "results",
    )
    stdev_pipeline.run()
    print("Stdev pipeline completed.")

if __name__ == "__main__":
    main()