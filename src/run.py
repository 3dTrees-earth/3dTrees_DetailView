import logging
import os
import sys
from pathlib import Path
from parameters import Parameters

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_las_files(directory):
    """Find all LAS/LAZ files in the given directory"""
    directory = Path(directory)
    las_files = list(directory.glob("*.las")) + list(directory.glob("*.laz"))
    if not las_files:
        raise FileNotFoundError(f"No LAS/LAZ files found in {directory}")
    return las_files


def main():
    """Main entry point for the 3D tree species classification tool"""
    try:
        # Parse parameters
        params = Parameters()
        logger.info(f"Starting DetailView with parameters: {params}")

        # Ensure output directory exists
        os.makedirs(params.output_dir, exist_ok=True)

        # Set threading parameters for optimal performance
        os.environ["OMP_NUM_THREADS"] = str(params.n_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(params.n_threads)
        os.environ["MKL_NUM_THREADS"] = str(params.n_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(params.n_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(params.n_threads)

        if params.mode.lower() == "predict":
            # Check if prediction_data is a directory or file
            prediction_path = Path(params.prediction_data)

            if prediction_path.is_dir():
                # If it's a directory, find LAS/LAZ files
                las_files = find_las_files(prediction_path)
                logger.info(
                    f"Found {len(las_files)} LAS/LAZ files in {prediction_path}"
                )

                # Use the first file found and set path_las to the directory
                first_file = las_files[0]
                logger.info(f"Using file: {first_file}")
                params.prediction_data = str(first_file)
                params.path_las = str(prediction_path)
            elif prediction_path.is_file():
                # If it's a file, use it directly
                logger.info(f"Using file: {prediction_path}")
            else:
                raise FileNotFoundError(f"Input path does not exist: {prediction_path}")

            # Run prediction
            logger.info("Running tree species prediction...")
            import predict

            outfile, outfile_probs, joined, data_probs_df = predict.run_predict(
                prediction_data=params.prediction_data,
                path_las=params.path_las,
                model_path=params.model_path,
                tree_id_col=params.tree_id_col,
                n_aug=params.n_aug,
                output_dir=str(params.output_dir),
                path_csv_train=params.path_csv_train,
                path_csv_lookup=params.path_csv_lookup,
            )

            logger.info(f"Prediction completed successfully!")
            logger.info(f"Results saved to: {outfile}")
            logger.info(f"Probabilities saved to: {outfile_probs}")
            logger.info(
                f"Predicted {len(joined)} trees across {joined['species'].nunique()} species"
            )

        elif params.mode.lower() == "train":
            # Run training
            logger.info("Starting model training...")
            import training

            logger.info("Training completed!")

        elif params.mode.lower() == "serve":
            # Start FastAPI server
            logger.info("Starting FastAPI server...")
            import uvicorn
            import main as fastapi_app

            uvicorn.run(fastapi_app.app, host="0.0.0.0", port=8000, log_level="info")

        else:
            logger.error(
                f"Unknown mode: {params.mode}. Supported modes: predict, train, serve"
            )
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
