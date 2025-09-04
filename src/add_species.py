"""

Merge PredInstance and Species Prediction

This script processes a single LAS file, merging species prediction data into the LAS file.
Functions:
    process_point_cloud(las_file, csv_folder, output_file):
        Processes a single LAS file by merging species prediction data from CSV files.
        Args:
            las_file (str): The path to the LAS file to process.
            csv_folder (str): The path to the folder containing the CSV files.
            output_file (str): The path to the output LAS file.
Usage:
    python merge_pred_species.py <las_file> <csv_folder> <output_file>
    <las_file> - The path to the LAS file to process.
    <csv_folder> - The path to the directory containing the CSV files.
    <output_file> - The path to the output LAS file.
"""

import pandas as pd
import laspy
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


def process_point_cloud(las_file, csv_folder, output_file):
    # Use environment variable for predictions file path
    predictions_path = os.environ.get(
        "OUTFILE", os.path.join(csv_folder, "predictions.csv")
    )

    # If no environment variable, try to find the most recent predictions file
    if not os.path.exists(predictions_path):
        import glob

        prediction_files = glob.glob(os.path.join(csv_folder, "predictions.csv"))
        if prediction_files:
            # Get the most recent file
            predictions_path = max(prediction_files, key=os.path.getctime)
            print(f"Found predictions file: {predictions_path}")
        else:
            print(f"No predictions file found in {csv_folder}")
            return

    if not os.path.exists(predictions_path):
        print(f"predictions file not found: {predictions_path}")
        return

    if not os.path.exists(las_file):
        print(f"LAS file not found: {las_file}")
        return

    # load the predictions data
    predictions = pd.read_csv(predictions_path)

    # print the LAS file
    las = laspy.read(las_file)

    # Use 'filename' column as the key and 'species_id' as the value
    attr_dict = predictions.set_index("filename")["species_id"].to_dict()

    print("Creating a new LAS file with the same header...")
    new_las = laspy.create(
        point_format=las.header.point_format, file_version=las.header.version
    )

    # copy the header from the original LAS file
    new_las.header = las.header

    # add the new attribute
    new_las.add_extra_dim(laspy.ExtraBytesParams(name="species_id", type=np.int32))

    # Print the data types of the new attributes
    print("New attributes added:")
    for dim in new_las.point_format.extra_dimensions:
        print(f"{dim.name}: {dim.dtype}")

    print("Copying all original point data...")
    for dimension in las.point_format.dimensions:
        if dimension.name == "species_id":
            continue
        setattr(new_las, dimension.name, getattr(las, dimension.name))

    print("Populating new attributes...")
    species_id = np.full(len(las.PredInstance), -1, dtype=np.int32)

    # Convert PredInstance to string and try different formats for matching
    for i, pred_instance in enumerate(las.PredInstance):
        pred_instance_str = str(pred_instance)

        # Try exact match first
        if pred_instance_str in attr_dict:
            species_id[i] = attr_dict[pred_instance_str]
        else:
            # Try with "tree_" prefix (most likely case)
            tree_key = f"tree_{pred_instance_str}"
            if tree_key in attr_dict:
                species_id[i] = attr_dict[tree_key]
            else:
                # Try removing "tree_" prefix if it exists
                if pred_instance_str.startswith("tree_"):
                    clean_key = pred_instance_str[5:]  # Remove "tree_" prefix
                    if clean_key in attr_dict:
                        species_id[i] = attr_dict[clean_key]

    new_las.species_id = species_id

    print(f"Saving the new LAS file to {output_file}...")
    new_las.write(output_file)


def main(las_file, csv_folder, output_file):
    process_point_cloud(las_file, csv_folder, output_file)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python merge_pred_species.py <las_file> <csv_folder> <output_file>"
        )
        sys.exit(1)

    las_file = sys.argv[1]
    csv_folder = sys.argv[2]
    output_file = sys.argv[3]
    main(las_file, csv_folder, output_file)
