import argparse
import os

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.images import resize_image, split_image_into_tiles


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "DIRPATH",
        type=str,
        help="Path in nnUNet format to get the images and their labels from or path to a directory with images.",
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
    parser.add_argument(
        "--is-nnunet-dir",
        action="store_true",
        help="Whether the input directory is in nnUNet format (default: False)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.is_nnunet_dir:
        images_path = os.path.join(args.DIRPATH, "imagesTr")
        labels_path = os.path.join(args.DIRPATH, "labelsTr")
    else:
        images_path = args.DIRPATH

    output_dir = args.OUTPUT_DIRPATH
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hdf5_file = h5py.File(
        os.path.join(output_dir, args.filename + f"_{args.modality}.hdf5"), "w"
    )
    print(f"Creating dataset to be translated at {hdf5_file}")
    modality = hdf5_file.create_group(args.modality)
    image_group = modality.create_group("images")
    if args.is_nnunet_dir:  # only create labels group if the input is in nnUNet format
        label_group = modality.create_group("labels")

    counter = 0
    for image_name in tqdm(os.listdir(images_path)):
        if os.path.isdir(os.path.join(images_path, image_name)):
            continue
        # Load the image and label
        image = np.array(
            Image.open(os.path.join(images_path, image_name)).convert("L")
        ).astype(np.uint8)
        image = resize_image(image, args.scale_factor)
        image_tiles, image_locations = split_image_into_tiles(image, args.tile_size, args.tile_size)

        # Load the label if the input is in nnUNet format
        if args.is_nnunet_dir:
            label_name = image_name.replace("_0000.png", ".png")
            label = np.array(
                Image.open(os.path.join(labels_path, label_name))
            ).astype(np.uint8)
            label = resize_image(label, args.scale_factor)
            label_tiles, _ = split_image_into_tiles(label, args.tile_size, args.tile_size)

        # Save the tiles and the corresponding labels to the HDF5 file
        for i, (tile, (x, y)) in enumerate(zip(image_tiles, image_locations)):
            image_dataset = image_group.create_dataset(
                str(counter),
                data=tile,
            )
            image_dataset.attrs["path_to_original"] = os.path.join(images_path, image_name)
            image_dataset.attrs["location"] = (x, y)

            # Create the label dataset if the input is in nnUNet format
            if args.is_nnunet_dir:
                label_dataset = label_group.create_dataset(
                    str(counter),
                    data=label_tiles[i],
                )
                label_dataset.attrs["path_to_original"] = os.path.join(labels_path, label_name)
                label_dataset.attrs["location"] = (x, y)
            counter += 1
