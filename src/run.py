import os
import sys
import time
from datetime import datetime
from parameters import Parameters
from predict import run_predict
import shutil


def main():
    """Main entry point for tree species prediction using Parameters."""

    # Record starting time
    start_time = time.time()
    start_datetime = datetime.now()
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load parameters from CLI arguments and environment variables
    params = Parameters()

    # Validate input file exists
    shutil.copy(params.dataset_path, "/in/input.laz")
    if not os.path.isfile(params.dataset_path):
        print(f"Input file not found: {params.dataset_path}")
        sys.exit(2)

    # Run prediction
    print(f"Running prediction on: {params.dataset_path}")
    print(f"Output directory: {params.output_dir}")

    run_predict(params)

    # Record end time and calculate elapsed time
    end_time = time.time()
    end_datetime = datetime.now()
    elapsed_time = end_time - start_time

    print("Processing complete!")
    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")


if __name__ == "__main__":
    main()
