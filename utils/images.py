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
