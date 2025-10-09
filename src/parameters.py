from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from typing import Literal


class Parameters(BaseSettings):
    """CLI parameters for tree species prediction"""

    dataset_path: str = Field(
        "/in",
        description="Path to input LAS/LAZ file for prediction",
        alias=AliasChoices("dataset-path", "dataset_path"),
    )

    path_las: str = Field(
        "/in",
        description="Alternative LAS path (optional)",
        alias=AliasChoices("path-las", "path_las"),
    )

    model_path: str = Field(
        "/app/model_ft_202412171652_3",
        description="Path to trained model file",
        alias=AliasChoices("model-path", "model_path"),
    )

    tree_id_col: str = Field(
        "PredInstance",
        description="Column name for tree IDs in LAS file",
        alias=AliasChoices("tree-id-col", "tree_id_col"),
    )

    n_aug: int = Field(
        10,
        description="Number of augmentation iterations",
        alias=AliasChoices("n-aug", "n_aug"),
    )

    output_dir: str = Field(
        "/output",
        description="Output directory for predictions",
        alias=AliasChoices("output-dir", "output_dir"),
    )

    path_csv_train: str = Field(
        "default_vals",
        description="Path to training CSV for height normalization",
        alias=AliasChoices("path-csv-train", "path_csv_train"),
    )

    path_csv_lookup: str = Field(
        "/app/lookup.csv",
        description="Path to species lookup CSV",
        alias=AliasChoices("path-csv-lookup", "path_csv_lookup"),
    )

    projection_backend: Literal["numpy", "torch"] = Field(
        "numpy",
        description="Backend for point cloud projection",
        alias=AliasChoices("projection-backend", "projection_backend"),
    )

    output_type: Literal["csv", "las", "both"] = Field(
        "csv",
        description="Output format type",
        alias=AliasChoices("output-type", "output_type"),
    )

    model_config = SettingsConfigDict(
        env_prefix="PREDICT_",
        cli_parse_args=True,
    )
