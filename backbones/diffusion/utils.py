# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import os
import shutil
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist


def copy_source(file: str, output_dir: str):
    """
    Copies a file to the output directory.

    Parameters
    ----------
    file : str
        The path to the file to copy.
    output_dir : str
        The path to the directory to copy the file to.
    """
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params: List[torch.nn.Parameter]):
    """
    Broadcasts the parameters to all GPUs.

    Parameters
    ----------
    params : List[torch.nn.Parameter]
        The parameters to broadcast.
    """
    for param in params:
        dist.broadcast(param.data, src=0)


def var_func_vp(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """
    Variance preserving schedule for the diffusion coefficients. From https://arxiv.org/pdf/2011.13456.pdf

    Parameters
    ----------
    t : torch.Tensor
        A tensor of time steps.
    beta_min : float
        The minimum beta.
    beta_max : float
        The maximum beta.

    Returns
    -------
    torch.Tensor
        The variance of the diffusion coefficients for the given time steps.
    """
    log_mean_coeff = -0.25 * t ** 2 * \
        (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """
    Geometric variance schedule for the diffusion coefficients.

    Parameters
    ----------
    t : torch.Tensor
        A tensor of time steps.
    beta_min : float
        The minimum beta.
    beta_max : float
        The maximum beta.

    Returns
    -------
    torch.Tensor
        The variance of the diffusion coefficients for the given time steps.
    """
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input: torch.Tensor, t: torch.Tensor, shape: List[int]) -> torch.Tensor:
    """
    Extracts the diffusion coefficients for the given time steps.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    t : torch.Tensor
        The time steps.
    shape : List[int]
        The shape of the tensor the coefficients will be broadcasted to.

    Returns
    -------
    out : torch.Tensor
        The diffusion coefficients for the given time steps.
    """
    # gather the diffusion coefficients for the given time steps
    out = torch.gather(input, 0, t)

    # reshape the diffusion coefficients to a shape that can be broadcasted
    # to the shape of the tensor the coefficients will be broadcasted to
    # For example: if shape is (B, C, H, W), the output will be (B, 1, 1, 1)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(num_timesteps: int, device: torch.device) -> torch.Tensor:
    """
    Generates a time schedule for the diffusion process.

    Parameters
    ----------
    num_timesteps : int
        The number of time steps.
    device : torch.device
        The device to run the computations on.

    Returns
    -------
    torch.Tensor
        The time schedule.
    """
    eps_small = 1e-3
    t = np.arange(0, num_timesteps + 1, dtype=np.float64)
    t = t / num_timesteps
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(
    n_timestep: int,
    beta_min: float,
    beta_max: float,
    device: torch.device,
    use_geometric: bool,
    eps_small=1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a noise schedule for a diffusion process, specifying the evolution of the variance across timesteps.
    This function supports both geometric and variance-preserving (VP) schedules.

    Parameters
    ----------
    n_timestep : int
        The total number of timesteps for the diffusion process.
    beta_min : float
        The minimum value for beta, determining the starting level of noise.
    beta_max : float
        The maximum value for beta, determining the peak level of noise.
    device : torch.device
        The computing device (CPU or GPU) on which tensors will be allocated.
    use_geometric : bool
        Flag to choose between a geometric or VP variance schedule.
    eps_small : float, optional
        A small constant to avoid division by zero, and to ensure numerical stability (default is 1e-3).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing tensors for the square root of betas, the square root of (1 - betas), and betas themselves.
    """

    # Normalize the timestep indices to [0, 1] scale and adjust by eps_small for stability
    t = np.arange(0, n_timestep + 1, dtype=np.float64) / n_timestep
    t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small

    # Calculate the variance for each timestep based on the selected schedule
    if use_geometric:
        # Geometric schedule function
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        # Variance-preserving schedule function
        var = var_func_vp(t, beta_min, beta_max)

    # Compute cumulative product of (1 - variance) to obtain alpha_bar_t
    alpha_bars = 1.0 - var

    # Calculate betas from alpha_bars to maintain the property beta_t = 1 - (alpha_bar_{t} / alpha_bar_{t-1})
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    # Prepend a very small beta at t=0 to avoid numerical issues in the first step
    first = torch.tensor(1e-8)
    betas = torch.cat(
        (first.unsqueeze(0),
         betas)
    ).to(device).type(torch.float32)

    # Compute the standard deviation of the noise to be added at each timestep
    sigmas = torch.sqrt(betas)

    # Compute the square root of alphas, which are used in the sampling process
    a_s = torch.sqrt(1 - betas)

    return sigmas, a_s, betas


def load_checkpoint(checkpoint_dir: str, netG, name_of_network: str, epoch: int, device: torch.device = 'cuda:0'):
    """
    Loads a checkpoint from a given directory.

    Parameters
    ----------
    checkpoint_dir : str
        The directory where the checkpoint is located.
    netG : torch.nn.Module
        The network to load the checkpoint into.
    name_of_network : str
        The name of the network.
    epoch : int
        The epoch to load the checkpoint from.
    device : torch.device, optional
        The device to load the checkpoint onto.
    """
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint

    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()
