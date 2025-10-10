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

    # Rename the input file to /in/input.laz for consistency
    target_input = "/in/input.laz"
    if os.path.abspath(params.dataset_path) != target_input:
        os.rename(params.dataset_path, target_input)
        params.dataset_path = target_input

    # Run prediction
    print(f"Running prediction on: {params.dataset_path}")
    print(f"Output directory: {params.output_dir}")

    run_predict(params)

    print("Processing complete!")


if __name__ == "__main__":
    main()
