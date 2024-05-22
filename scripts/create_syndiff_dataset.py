import argparse
import os
from typing import Tuple

import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.images import resize_image


def parse_args():
    """
    Parses command-line arguments.

    Returns
    -------
    argparse.Namespace
        A namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create HDF5 dataset compatible with the syndiff pipeline."
    )
    parser.add_argument(
        "--modality1_paths",
        nargs="+",
        required=True,
        type=str,
        help="Directories to search for Modality1 files. Subdirectories will also be searched.",
    )
    parser.add_argument(
        "--modality2_paths",
        nargs="+",
        required=True,
        type=str,
        help="Directories to search for Modality2 files. Subdirectories will also be searched.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The directory to save the output files.",
    )
    parser.add_argument(
        "--modality1",
        type=str,
        default="sem",
        help="The name of the first modality. Default is 'sem'.",
    )
    parser.add_argument(
        "--modality2",
        type=str,
        default="tem",
        help="The name of the second modality. Default is 'tem'.",
    )
    parser.add_argument(
        "--training_set_ratio",
        type=float,
        default=0.8,
        help="The proportion of the dataset to include in the train split. Default is 0.8.",
    )
    parser.add_argument(
        "--validation_set_ratio",
        type=float,
        default=0.1,
        help="The proportion of the dataset to include in the validation split. Default is 0.1.",
    )
    parser.add_argument(
        "--test_set_ratio",
        type=float,
        default=0.1,
        help="The proportion of the dataset to include in the test split. Default is 0.1.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="dataset",
        help="The name of the output HDF5 file. Default is 'dataset'.",
    )
    parser.add_argument(
        "--modality1_scale_factors",
        nargs="+",
        type=float,
        default=None,
        help="The factors by which to multiply the dimensions of the Modality1 images. Accepts multiple values. Default is 1.0.",
    )
    parser.add_argument(
        "--modality2_scale_factors",
        nargs="+",
        type=float,
        default=None,
        help="The factors by which to multiply the dimensions of the Modality2 images. Accepts multiple values. Default is 1.0.",
    )
    parser.add_argument(
        "--subdirectories_to_exclude_from_modality1",
        nargs="+",
        type=str,
        default=("derivatives", ".git"),
        help="The subdirectories to exclude from the Modality1 search. Default is derivatives and .git.",
    )
    parser.add_argument(
        "--subdirectories_to_exclude_from_modality2",
        nargs="+",
        type=str,
        default=("derivatives", ".git"),
        help="The subdirectories to exclude from the Modality2 search. Default is derivatives and .git.",
    )
    return parser.parse_args()


def find_png_files(directory: str, excluded_dirs: Tuple[str] = ("derivatives", ".git")):
    """
    Finds all image files in a directory and its subdirectories but excludes certain
    directories.

    Parameters
    ----------
    directory : str
        The directory to search for image files.
    excluded_dirs : tuple of str
        Directories to exclude from the search.

    Returns
    -------
    list of str
        A list of paths to the image files.
    """
    png_files = []
    for root, dirs, files in os.walk(directory):
        for excluded_dir in excluded_dirs:
            if excluded_dir in dirs:
                dirs.remove(excluded_dir)  # don't visit excluded directories
        for file in files:
            if file.endswith(".png") or file.endswith(".tif"):
                png_files.append(os.path.join(root, file))
    return png_files


def split_data(
    data,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
):
    """
    Splits the data into train, validation, and test sets.

    Parameters
    ----------
    data : ndarray
        The data to split.
    train_size : float
        The proportion of the dataset to include in the train split.
    val_size : float
        The proportion of the dataset to include in the validation split.
    test_size : float
        The proportion of the dataset to include in the test split.
    random_state : int
        The seed used by the random number generator.

    Returns
    -------
    dict
        A dictionary containing the train, validation, and test sets.
    """
    assert train_size + val_size + test_size == 1, "The sizes must sum up to 1."
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    relative_val_size = val_size / (train_size + val_size)
    train_data, val_data = train_test_split(
        train_data, test_size=relative_val_size, random_state=random_state
    )
    return {"train": train_data, "val": val_data, "test": test_data}


if __name__ == "__main__":
    args = parse_args()

    if args.modality1_scale_factors is None:
        args.modality1_scale_factors = [1.0] * len(args.modality1_paths)
    if args.modality2_scale_factors is None:
        args.modality2_scale_factors = [1.0] * len(args.modality2_paths)

    # Ensure the number of scale factors matches the number of paths
    assert len(args.modality1_paths) == len(
        args.modality1_scale_factors
    ), f"The number of {args.modality1} (Modality 1) paths must match the number of {args.modality1} scale factors."
    assert len(args.modality2_paths) == len(
        args.modality2_scale_factors
    ), f"The number of {args.modality2} (Modality 2) paths must match the number of {args.modality2} scale factors."

    modality1_data = [
        {
            "paths": split_data(
                find_png_files(
                    modality1_directory, args.subdirectories_to_exclude_from_modality1
                ),
                args.training_set_ratio,
                args.validation_set_ratio,
                args.test_set_ratio,
            ),
            "scale factor": scale_factor,
        }
        for modality1_directory, scale_factor in zip(
            args.modality1_paths, args.modality1_scale_factors
        )
    ]
    modality2_data = [
        {
            "paths": split_data(
                find_png_files(
                    modality2_directory, args.subdirectories_to_exclude_from_modality2
                ),
                args.training_set_ratio,
                args.validation_set_ratio,
                args.test_set_ratio,
            ),
            "scale factor": scale_factor,
        }
        for modality2_directory, scale_factor in zip(
            args.modality2_paths, args.modality2_scale_factors
        )
    ]
    data = {args.modality1: modality1_data, args.modality2: modality2_data}

    output_dir = args.output_path

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the data to the HDF5 file
    for split in ["train", "val", "test"]:
        # Create a new HDF5 file for each split
        hdf5_file = h5py.File(
            os.path.join(output_dir, args.filename + "_" + split + ".hdf5"), "w"
        )
        for modality in [args.modality1, args.modality2]:
            modality_group = hdf5_file.create_group(modality)
            counter = 0
            for dataset_data in data[modality]:
                for image_path in tqdm(dataset_data["paths"][split]):
                    image_array = np.array(Image.open(image_path).convert("L")).astype(
                        np.uint8
                    )

                    num_values = len(np.unique(image_array))
                    assert (
                        num_values > 1
                    ), f"Image {image_path} contains only one unique value (It's probably blank)."
                    assert (
                        num_values > 3
                    ), f"Image {image_path} contains only three unique values (It's probably a segmenation map (background, axon, myelin))."

                    modality_group.create_dataset(
                        str(counter),
                        data=resize_image(image_array, dataset_data["scale factor"]),
                    )
                    counter += 1

    hdf5_file.close()
    print(f"Dataset successfully saved to {output_dir}")
