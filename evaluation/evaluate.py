from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from backbones.diffusion.reverse_process import sample_from_model
from utils.utils import to_range_0_1


def l1(fake_sample: np.ndarray, real_sample: np.ndarray) -> float:
    """
    Compute the L1 loss between two numpy arrays.

    Parameters
    ----------
    fake_sample : np.ndarray
        The numpy array representing the generated or predicted sample.
    real_sample : np.ndarray
        The numpy array representing the ground truth or actual sample.

    Returns
    -------
    float
        The computed L1 loss, which is the mean of the absolute differences between the two arrays.
    """
    return np.mean(np.abs(fake_sample - real_sample))


class Evaluator:
    def __init__(
        self,
        num_timesteps: int,
        time_schedule: torch.Tensor,
        latent_dim: int,
        device: torch.device,
        paired: bool = True,
    ):
        """
        Initialize the evaluator

        Parameters
        ----------
        num_timesteps : int
            The number of timesteps to use for the diffusion process.
        time_schedule : torch.Tensor
            The time schedule to use for the diffusion process.
        latent_dim : int
            The latent dimension of the model.
        device : torch.device
            The device to use for the evaluation.
        paired : bool
            Whether the evaluation is for a paired dataset.
        """
        self.num_timesteps = num_timesteps
        self.device = device
        self.time_schedule = time_schedule
        self.latent_dim = latent_dim
        self.paired = paired

    def update(
        self,
        results: Dict[str, Dict[str, List[float]]],
        images_modality1: torch.Tensor,
        images_modality2: torch.Tensor,
        gen_diffusive_1: nn.Module,
        gen_diffusive_2: nn.Module,
        pos_coeff: nn.Module
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Update the evaluator

        Parameters
        ----------
        images_modality1 : torch.Tensor
            The images from modality 1.
        images_modality2 : torch.Tensor
            The images from modality 2.
        gen_diffusive_1 : nn.Module
            The generator for modality 1.
        gen_diffusive_2 : nn.Module
            The generator for modality 2.
        pos_coeff : nn.Module
            The positive coefficient.

        Returns
        -------
        Dict[str, Dict[str, List[float]]]
            The results.
        """
        images_dim = images_modality1.shape

        # Modality 2 -> Modality 1
        x1_t = torch.cat((torch.randn(images_dim, device=self.device), images_modality2), axis=1)
        fake_sample2_1 = sample_from_model(
            pos_coeff,
            gen_diffusive_1,
            self.num_timesteps,
            x1_t,
            self.time_schedule,
            latent_dimension=self.latent_dim
        )

        # Modality 1 -> Modality 2
        x2_t = torch.cat((torch.randn(images_dim, device=self.device), images_modality1), axis=1)
        fake_sample1_2 = sample_from_model(
            pos_coeff,
            gen_diffusive_2,
            self.num_timesteps,
            x2_t,
            self.time_schedule,
            latent_dimension=self.latent_dim
        )

        if not self.paired:
            # Modality 2 -> Modality 1 -> Modality 2
            x21_t = torch.cat((torch.randn(images_dim, device=self.device), fake_sample2_1), axis=1)
            fake_sample2_1_2 = sample_from_model(
                pos_coeff,
                gen_diffusive_2,
                self.num_timesteps,
                x21_t,
                self.time_schedule,
                latent_dimension=self.latent_dim
            )

            # Modality 1 -> Modality 2 -> Modality 1
            x12_t = torch.cat((torch.randn(images_dim, device=self.device), fake_sample1_2), axis=1)
            fake_sample1_2_1 = sample_from_model(
                pos_coeff,
                gen_diffusive_1,
                self.num_timesteps,
                x12_t,
                self.time_schedule,
                latent_dimension=self.latent_dim
            )
            fake_sample1_2_1 = to_range_0_1(fake_sample1_2_1).cpu().numpy()
            fake_sample2_1_2 = to_range_0_1(fake_sample2_1_2).cpu().numpy()

        fake_sample2_1 = to_range_0_1(fake_sample2_1).cpu().numpy()
        fake_sample1_2 = to_range_0_1(fake_sample1_2).cpu().numpy()

        real_sample1 = to_range_0_1(images_modality1).cpu().numpy()
        real_sample2 = to_range_0_1(images_modality2).cpu().numpy()

        if self.paired:
            results['psnr']['1_2'].append(float(psnr(fake_sample1_2, real_sample2)))
            results['psnr']['2_1'].append(float(psnr(fake_sample2_1, real_sample1)))
            results['ssim']['1_2'].append(float(ssim(fake_sample1_2[0, 0], real_sample2[0, 0], data_range=1)))
            results['ssim']['2_1'].append(float(ssim(fake_sample2_1[0, 0], real_sample1[0, 0], data_range=1)))
            results['l1']['1_2'].append(float(l1(fake_sample1_2, real_sample2)))
            results['l1']['2_1'].append(float(l1(fake_sample2_1, real_sample1)))
        else:
            results['psnr']['1_2_1'].append(float(psnr(fake_sample1_2_1, real_sample1)))
            results['psnr']['2_1_2'].append(float(psnr(fake_sample2_1_2, real_sample2)))
            results['ssim']['1_2_1'].append(float(ssim(fake_sample1_2_1[0, 0], real_sample1[0, 0], data_range=1)))
            results['ssim']['2_1_2'].append(float(ssim(fake_sample2_1_2[0, 0], real_sample2[0, 0], data_range=1)))
            results['l1']['1_2_1'].append(float(l1(fake_sample1_2_1, real_sample1)))
            results['l1']['2_1_2'].append(float(l1(fake_sample2_1_2, real_sample2)))

        return results
