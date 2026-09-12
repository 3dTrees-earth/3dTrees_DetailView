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
        "/app/model_europe_v1",
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

    output_species_id_dim: str = Field(
        "species_id",
        description="Name of the LAS extra dimension used for predicted species IDs",
        alias=AliasChoices("output-species-id-dim", "output_species_id_dim"),
    )

    output_species_prob_dim: str = Field(
        "species_prob",
        description="Name of the LAS extra dimension used for predicted species probabilities",
        alias=AliasChoices("output-species-prob-dim", "output_species_prob_dim"),
    )

    batch_size: int | None = Field(
        None,
        description="Inference batch size. Larger values improve GPU throughput but use more VRAM; try 10-32 on production CUDA/HPC nodes.",
        alias=AliasChoices("batch-size", "batch_size"),
    )

    num_workers: int | None = Field(
        None,
        description="Number of DataLoader worker processes. More workers hide CPU projection latency but use more RAM; try 4-8 on production CUDA/HPC nodes.",
        alias=AliasChoices("num-workers", "num_workers"),
    )

    pin_memory: bool | None = Field(
        None,
        description="Whether DataLoader should use pinned host memory. Enable for CUDA/HPC inference to speed host-to-GPU transfers.",
        alias=AliasChoices("pin-memory", "pin_memory"),
    )

    early_stop_aug: bool = Field(
        False,
        description="Stop augmentation passes early when predictions stabilize. Off by default to preserve fixed n_aug behavior.",
        alias=AliasChoices("early-stop-aug", "early_stop_aug"),
    )

    early_stop_min_aug: int = Field(
        3,
        description="Minimum augmentation passes before early stopping can trigger.",
        alias=AliasChoices("early-stop-min-aug", "early_stop_min_aug"),
    )

    early_stop_patience: int = Field(
        2,
        description="Stable augmentation epochs required before stopping early.",
        alias=AliasChoices("early-stop-patience", "early_stop_patience"),
    )

    early_stop_change_threshold: int = Field(
        0,
        description="Maximum class changes vs. previous epoch still considered stable.",
        alias=AliasChoices(
            "early-stop-change-threshold", "early_stop_change_threshold"
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="PREDICT_",
        cli_parse_args=True,
    )
