import os
import sys
from datetime import datetime
from parameters import Parameters
from predict import run_predict


def main():
    """Main entry point for tree species prediction using Parameters."""

    # Load parameters from CLI arguments and environment variables
    params = Parameters()

    # Validate input file exists
    if not os.path.isfile(params.dataset_path):
        print(f"Input file not found: {params.dataset_path}")
        sys.exit(2)

    # Rename the input file to input.laz if needed
    if not params.dataset_path.endswith("input.laz"):
        input_dir = os.path.dirname(params.dataset_path)
        new_path = os.path.join(input_dir, "input.laz")
        os.rename(params.dataset_path, new_path)
        params.dataset_path = new_path
        print(f"Renamed input to: {params.dataset_path}")

    # Run prediction
    print(f"Running prediction on: {params.dataset_path}")
    print(f"Output directory: {params.output_dir}")

    run_predict(params)

    print("Processing complete!")


if __name__ == "__main__":
    main()
