from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from pathlib import Path


class Parameters(BaseSettings):
    """CLI parameters for 3D tree species classification tool"""

    # Input data parameters
    prediction_data: str = Field(
        "/input/circle_3_segmented.las",
        description="Path to input LAS/LAZ file or directory containing point cloud data",
        alias=AliasChoices("prediction-data", "prediction_data", "input"),
    )
    path_las: str = Field(
        "",
        description="Base path for LAS files when using CSV input",
        alias=AliasChoices("path-las", "path_las"),
    )
    tree_id_col: str = Field(
        "TreeID",
        description="Column name for tree IDs in the point cloud data",
        alias=AliasChoices("tree-id-col", "tree_id_col"),
    )

    # Model parameters
    model_path: str = Field(
        "./model_ft_202412171652_3",
        description="Path to the trained model file",
        alias=AliasChoices("model-path", "model_path"),
    )
    n_aug: int = Field(
        10,
        description="Number of augmentations for prediction (higher = more accurate but slower)",
        alias=AliasChoices("n-aug", "n_aug", "augmentations"),
        ge=1,
        le=100,
    )

    # Output parameters
    output_dir: Path = Field(
        "/output",
        description="Output directory for prediction results",
        alias=AliasChoices("output-dir", "output_dir", "output"),
    )

    # Lookup and training data
    path_csv_lookup: str = Field(
        "./lookup.csv",
        description="Path to species lookup CSV file",
        alias=AliasChoices("path-csv-lookup", "path_csv_lookup", "lookup"),
    )
    path_csv_train: str = Field(
        "default_vals",
        description="Path to training CSV file (for height normalization)",
        alias=AliasChoices("path-csv-train", "path_csv_train", "train-csv"),
    )

    # Processing parameters
    n_threads: int = Field(
        10,
        description="Number of threads for processing",
        alias=AliasChoices("n-threads", "n_threads", "threads"),
        ge=1,
        le=64,
    )

    # Model architecture parameters (usually don't change)
    n_class: int = Field(
        33,
        description="Number of tree species classes",
        alias=AliasChoices("n-class", "n_class"),
    )
    n_view: int = Field(
        7,
        description="Number of views for classification",
        alias=AliasChoices("n-view", "n_view"),
    )
    resolution: int = Field(
        256,
        description="Image resolution for processing",
        alias=AliasChoices("resolution", "res"),
    )
    batch_size: int = Field(
        1,
        description="Batch size for processing",
        alias=AliasChoices("batch-size", "batch_size"),
        ge=1,
        le=16,
    )

    # Mode selection
    mode: str = Field(
        "predict",
        description="Operation mode: predict, train, or serve",
        alias=AliasChoices("mode"),
    )

    model_config = SettingsConfigDict(
        case_sensitive=False, 
        cli_parse_args=True, 
        cli_ignore_unknown_args=True,
        env_prefix="DETAILVIEW_"
    )
