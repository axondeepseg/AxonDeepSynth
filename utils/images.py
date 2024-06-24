from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def resize_image(image: np.ndarray, scale_factor: float = 1.0):
    """
    Resizes an image by a specified factor.

    Parameters
    ----------
    image : ndarray
        The image to resize.
    scale_factor : float
        The factor by which to multiply the dimensions of the image. Default is 1.0.

    Returns
    -------
    ndarray
        The resized image.
    """
    if scale_factor != 1.0:
        image = (
            F.interpolate(
                torch.tensor(image).unsqueeze(0).unsqueeze(0),
                scale_factor=(scale_factor, scale_factor),
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .numpy()
        )
    return image


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
    list of tuples
        A list of tuples, each representing the location of a tile in the image.
    """
    tiles = []
    locations = []
    img_height, img_width = image_array.shape

    for y in range(0, img_height, tile_height):
        for x in range(0, img_width, tile_width):
            # Adjust the start position of the last tile in a row/column if necessary
            start_y = min(y, max(0, img_height - tile_height))
            start_x = min(x, max(0, img_width - tile_width))

            # Extract the tile
            tile = image_array[
                start_y: start_y + tile_height, start_x: start_x + tile_width
            ]
            tiles.append(tile)
            locations.append((start_x, start_y))

    return tiles, locations


def reconstruct_image_from_patches(
    patch_info_list: List[Dict[str, Union[str, List[int]]]],
    patch_size: Tuple[int, int],
    dir_key: str = "patch_dir",
    location_key: str = "image_location"
) -> np.ndarray:
    """
    Reconstruct an image from a list of patches.

    Parameters
    ----------
    patch_info_list : List[Dict[str, Union[str, List[int]]]],
        The list of patches to reconstruct the image from.
    patch_size : Tuple[int, int],
        The size of the patches.
    dir_key : str,
        The key in the patch info dict that contains the patch directory.
    location_key : str,
        The key in the patch info dict that contains the patch location.

    Returns
    -------
    np.ndarray
        The reconstructed image.
    """
    image_locations = np.vstack([np.asarray(patch_info[location_key]) for patch_info in patch_info_list])
    image_dirs = [patch_info[dir_key] for patch_info in patch_info_list]
    full_image = np.zeros((image_locations[-1, 1] + patch_size[1], image_locations[-1, 0] + patch_size[0]))
    for image_dir, image_location in zip(image_dirs, image_locations):
        patch = cv2.imread(image_dir, cv2.IMREAD_UNCHANGED)
        full_image[image_location[1]:image_location[1] + patch_size[1], image_location[0]:image_location[0] + patch_size[0]] = patch

    return full_image.astype(np.uint8)
