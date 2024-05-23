import argparse
import os

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.images import resize_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "DIRPATH",
        type=str,
        help="Path in nnUNet format to get the images and their labels from",
    )
    parser.add_argument(
        "OUTPUT_DIRPATH",
        type=str,
        help="Path to the directory where the new tiled up dataset will be saved",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1.0,
        help="The factor by which to multiply the dimensions of the images. Default is 1.0.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="The size of each tile (default: 512)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="tem",
        help="The name of the modality (default: tem)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="dataset",
        help="The name of the HDF5 file to be created (default: dataset)",
    )
    return parser.parse_args()


def split_image_into_tiles(image_array: np.ndarray, tile_height: int, tile_width: int):
    """
    Splits an image into tiles of specified size. If the image cannot be evenly divided,
    the last tile in a row or column will overlap with the previous tile.

    Parameters
    ----------
    image_array : ndarray
        Numpy array of the image.
    tile_height : int
        Height of each tile.
    tile_width : int
        Width of each tile.

    Returns
    -------
    list of ndarray
        A list of numpy arrays, each representing a tile.
    """
    tiles = []
    img_height, img_width = image_array.shape

    for y in range(0, img_height, tile_height):
        for x in range(0, img_width, tile_width):
            # Adjust the start position of the last tile in a row/column if necessary
            start_y = min(y, max(0, img_height - tile_height))
            start_x = min(x, max(0, img_width - tile_width))

            # Extract the tile
            tile = image_array[
                start_y : start_y + tile_height, start_x : start_x + tile_width
            ]
            tiles.append(tile)

    return tiles


if __name__ == "__main__":
    args = parse_args()

    train_images_path = os.path.join(args.DIRPATH, "imagesTr")
    train_labels_path = os.path.join(args.DIRPATH, "labelsTr")

    output_dir = args.OUTPUT_DIRPATH
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hdf5_file = h5py.File(
        os.path.join(output_dir, args.filename + f"_{args.modality}.hdf5"), "w"
    )
    modality = hdf5_file.create_group(args.modality)
    image_group = modality.create_group("images")
    label_group = modality.create_group("labels")

    counter = 0
    for train_image in tqdm(os.listdir(train_images_path)):
        # Load the image and label
        train_label = train_image.replace("_0000.png", ".png")
        image = np.array(
            Image.open(os.path.join(train_images_path, train_image)).convert("L")
        ).astype(np.uint8)
        label = np.array(
            Image.open(os.path.join(train_labels_path, train_label))
        ).astype(np.uint8)

        # Preprocess the image and label
        image = resize_image(image, args.scale_factor)
        label = resize_image(label, args.scale_factor)

        # Split the image and label into tiles
        image_tiles = split_image_into_tiles(image, args.tile_size, args.tile_size)
        label_tiles = split_image_into_tiles(label, args.tile_size, args.tile_size)

        # Save the tiles and the corresponding labels to the HDF5 file
        for tile, label_tile in zip(image_tiles, label_tiles):
            image_group.create_dataset(
                str(counter),
                data=tile,
            )
            label_group.create_dataset(
                str(counter),
                data=label_tile,
            )
            counter += 1
