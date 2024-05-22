import os
from typing import Callable, List, Literal, Optional

import h5py
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize the input tensor to the range [-1, 1].
    This cannot be a lambda function because it is not serializable.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be normalized.

    Returns
    -------
    torch.Tensor
        Normalized tensor.
    """
    return (x / 127.5) - 1.0


def add_channel_dimension(x: torch.Tensor) -> torch.Tensor:
    """
    Add a channel dimension to the input tensor.
    This cannot be a lambda function because it is not serializable.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to which the channel dimension should be added.

    Returns
    -------
    torch.Tensor
        Tensor with an additional channel dimension.
    """
    return x.unsqueeze(0)


def CreateDatasetSynthesis(
    phase: Literal["train", "val", "test"],
    input_path: str,
    contrast1: str = "sem",
    contrast2: str = "tem",
    size: int = 512,
    paired: bool = False,
    **kwargs,
):
    """
    Create a dataset for the synthesis task from the specified phase of the dataset.

    Parameters
    ----------
    phase : Literal["train", "val", "test"]
        The phase of the dataset to use. Must be one of "train", "val", or "test".
    input_path : str
        The path to the directory containing the dataset files.
    contrast1 : str, optional
        The name of the first contrast modality. Default is "sem".
    contrast2 : str, optional
        The name of the second contrast modality. Default is "tem".
    size : int, optional
        The size to which the images should be resized. Default is 512.
    paired : bool, optional
        Whether the dataset should be treated as paired. If True, the dataset will be treated
        as a paired dataset, where both modalities have the same number of samples and the element 
        at each index is paired with the element at the same index in the other modality.
        Default is False.
    kwargs : dict, optional
        Additional keyword arguments to pass to the dataset class.

    Returns
    -------
    DualModalityDataset
        A dataset object for the synthesis task.
    """
    target_file = os.path.join(input_path, "dataset_{}.hdf5".format(phase))
    if phase == "train":
        transformations = [
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Lambda(normalize_image),
            transforms.Lambda(add_channel_dimension),
        ]
    else:
        transformations = [
            transforms.RandomCrop(size),
            transforms.Lambda(normalize_image),
            transforms.Lambda(add_channel_dimension),
        ]

    dataset = DualModalityDataset(
        target_file,
        modality1=contrast1,
        modality2=contrast2,
        transformations=transformations,
        paired=paired
    )
    return dataset


class DualModalityDataset(Dataset):
    def __init__(
        self,
        path_hdf5_dataset: str,
        modality1: str = "sem",
        modality2: str = "tem",
        transformations: Optional[List[Callable]] = None,
        paired: bool = False,
    ):
        """
        Initialize a dataset designed to handle dual-modality data stored in an HDF5 file.
        This dataset is intended for use with two distinct data modalities (e.g., SEM and TEM images)
        that are stored in a hierarchical structure within the HDF5 file. At the top level, the HDF5
        file contains keys corresponding to the two modalities. Each modality key then contains datasets
        indexed by sequential integers as strings ('0', '1', '2', ..., 'len_modality'), representing
        individual samples.

        Parameters
        ----------
        path_hdf5_dataset : str
            Path to the HDF5 dataset file. The file should have a top-level structure with keys
            corresponding to the two modalities, each containing datasets named by sequential integers
            representing individual samples.
        modality1 : str, optional
            The name of the first modality. Default is "sem".
        modality2 : str, optional
            The name of the second modality. Default is "tem".
        transformations : Optional[List[Callable]], optional
            A list of transformation functions to be applied to the data samples. Each function should
            take a single argument (the data sample) and return the transformed sample. If None, no
            transformations are applied. Default is None.
        paired : bool, optional
            Whether the dataset should be treated as paired. If True, the dataset will be treated
            as a paired dataset, where both modalities have the same number of samples and the element 
            at each index is paired with the element at the same index in the other modality.
            Default is False.
        """
        self.path_hdf5_dataset = path_hdf5_dataset
        self.transformations = transformations
        self.paired = paired
        self.modalities = [modality1, modality2]

        # Ensure there is a way to handle different lengths, e.g., cycle the shorter dataset
        with h5py.File(self.path_hdf5_dataset, "r") as file:
            assert all(
                [modality in list(file.keys()) for modality in self.modalities]
            ), f"The dataset should contain both modalities: {self.modalities}."
            assert (
                len(self.modalities) == 2
            ), "The dataset should contain exactly two modalities."
            print(f"Modalities: {self.modalities}")
            self.len1 = len(file[self.modalities[0]])
            self.len2 = len(file[self.modalities[1]])
            if self.paired:
                assert self.len1 == self.len2, "The two modalities should have the same length when paired."

        # Store the maximum length out of the two modalities which will be the dataset length
        self.max_len = max(self.len1, self.len2)

        # Store the indices for shuffling
        self.indices1 = np.arange(self.len1)
        self.indices2 = np.arange(self.len2)

    def __len__(self):
        """
        Returns the length of the dataset, which is determined by the longer of the two modalities.

        Returns
        -------
        int
            The length of the dataset.
        """
        return self.max_len

    def __getitem__(self, idx: int):
        """
        Retrieves a sample from the dataset at the specified index. If the dataset indices for a modality
        are exceeded (in case of different lengths of modalities), the indexing wraps around to simulate
        a cyclic dataset. This method ensures that each call retrieves a pair of samples, one from each
        modality, applying any specified transformations before returning them.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the two samples from the dataset, potentially transformed if transformations
            were specified.
        """
        idx1 = self.indices1[idx % self.len1]
        idx2 = self.indices2[idx % self.len2]

        # HDF5 file is opened in __getitem__ to ensure compatibility with DataLoader multiprocessing
        with h5py.File(self.path_hdf5_dataset, "r") as file:
            sample1, sample2 = torch.from_numpy(
                np.array(file[self.modalities[0]][str(idx1)])
            ), torch.from_numpy(np.array(file[self.modalities[1]][str(idx2)]))

            # Apply transformations
            if self.transformations:
                for transform in self.transformations:
                    sample1 = transform(sample1)
                    sample2 = transform(sample2)

        return sample1, sample2

    def on_epoch_start(self):
        """
        Shuffles the indices for each modality at the start of each epoch. This method should be called
        manually at the beginning of each epoch if the dataset is used in a training loop. This is only 
        necessary if the dataset is not paired.
        """
        if not self.paired:
            np.random.shuffle(self.indices1)
            np.random.shuffle(self.indices2)


class DatasetToBeTranslated(Dataset):
    def __init__(
        self,
        path_hdf5_dataset: str,
        modality: str = "sem",
    ):
        """
        Initializes the dataset object for a specific modality within an HDF5 file.
        The structure of the HDF5 file is expected to have the modality as the first level,
        under which there are "images" and "labels" datasets. Both "images" and "labels"
        contain data indexed by sequential integers as strings ('0', '1', '2', ...).

        Parameters
        ----------
        path_hdf5_dataset : str
            The file path to the HDF5 dataset.
        modality : str, optional
            The modality to be accessed within the HDF5 dataset, by default "sem".
        """
        self.path_hdf5_dataset = path_hdf5_dataset
        self.modality = modality
        self.image_transforms = [
            transforms.Lambda(lambda x: (x / 127.5) - 1.0),
            transforms.Lambda(lambda x: x.unsqueeze(0)),
        ]
        self.label_transforms = [
            transforms.Lambda(lambda x: x.unsqueeze(0)),
        ]

        # Store the length of the dataset
        with h5py.File(self.path_hdf5_dataset, "r") as file:
            assert self.modality in list(
                file.keys()
            ), f"The dataset should contain the modality: {self.modality}."
            self.len = len(file[self.modality]["images"])

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return self.len

    def __getitem__(self, idx: int):
        """
        Retrieves an image and its corresponding label from the dataset at the specified index.
        Applies any specified transformations before returning them.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the image and label from the dataset, potentially transformed.
        """
        with h5py.File(self.path_hdf5_dataset, "r") as file:
            image, label = torch.from_numpy(
                np.array(file[self.modality]["images"][str(idx)])
            ), torch.from_numpy(np.array(file[self.modality]["labels"][str(idx)]))

            # Apply transformations
            for transform in self.image_transforms:
                image = transform(image)

            for transform in self.label_transforms:
                label = transform(label)

        return image, label
