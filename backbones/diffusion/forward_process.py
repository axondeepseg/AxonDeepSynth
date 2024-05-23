# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import Tuple

import numpy as np
import torch

from backbones.diffusion.utils import extract, get_sigma_schedule


class Diffusion_Coefficients():
    def __init__(self, n_timestep: int, beta_min: float, beta_max: float, device: torch.device, use_geometric: bool):
        """
        Initialize the diffusion coefficients.

        Parameters
        ----------
        n_timestep : int
            The number of time steps.
        beta_min : float
            The minimum beta.
        beta_max : float
            The maximum beta.
        device : torch.device
            The device to run the computations on.
        use_geometric : bool
            Whether to use geometric or variance preserving schedule.
        """
        # Generate the schedule for the diffusion process
        self.sigmas, self.a_s, _ = get_sigma_schedule(
            n_timestep, beta_min, beta_max, device=device, use_geometric=use_geometric)

        # Cumulative product of the diffusion coefficients alpha_bar
        self.a_s_cum = np.cumprod(self.a_s.cpu())

        # Standard deviation of q(x_t | x_0) ~ N(sqrt(alpha_bar)x_0, (1-alpha_bar^2)I)
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)

        # Move to the device
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)


def q_sample(coeff: Diffusion_Coefficients, x_start: torch.Tensor, t: torch.Tensor, *, noise: torch.Tensor = None) -> torch.Tensor:
    """
    Simulates the forward diffusion process to generate a noisy version of the input data x_start at a specified timestep t.
    This function progressively adds Gaussian noise to the data based on the diffusion coefficients.
    q(x_t | x_0) ~ N(sqrt(alpha_bar)x_0, (1-alpha_bar^2)I)

    Parameters
    ----------
    coeff : Diffusion_Coefficients
        An object containing the precomputed cumulative products of scaling factors (a_s_cum) and noise levels (sigmas_cum) for each timestep.
    x_start : torch.Tensor
        The initial data before any noise has been added.
    t : torch.Tensor
        The timestep at which the diffusion state is to be computed, with 0 indicating the initial state and increasing timesteps representing further diffusion.
    noise : torch.Tensor, optional
        A tensor of noise matching the dimensions of x_start. If not provided, Gaussian noise is generated.

    Returns
    -------
    torch.Tensor
        The diffused data at timestep t, represented as x_t. This data is a mix of the scaled original data and added noise.
    """
    # Generate random Gaussian noise if none is provided
    if noise is None:
        noise = torch.randn_like(x_start)

    # Calculate the diffused data for the specified timestep t using precomputed coefficients:
    # a_s_cum[t] scales down the original data to maintain a proportion of the original variance
    # sigmas_cum[t] determines the standard deviation of the noise to be added at timestep t
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
        extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff: Diffusion_Coefficients, x_start: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a pair of consecutively diffused images (x_t and x_{t+1}) from the initial data x_start at a specified timestep t.
    This function is used to create training pairs for models that need to understand the transition dynamics between subsequent timesteps in a diffusion process.

    Parameters
    ----------
    coeff : Diffusion_Coefficients
        An object containing the diffusion coefficients, which include precomputed scaling factors (a_s) and noise levels (sigmas) for each timestep.
    x_start : torch.Tensor
        The initial data (x_0) before any noise has been added.
    t : int
        The timestep at which the first diffusion state is to be computed. The function also computes the state for the next timestep (t+1).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing the diffused data at timestep t (x_t) and t+1 (x_{t+1}). These pairs are used to train the model on how data evolves through the diffusion process.
    """
    # Generate random Gaussian noise for the second diffusion step
    noise = torch.randn_like(x_start)

    # Compute the diffused state x_t using the precomputed coefficients at timestep t
    x_t = q_sample(coeff, x_start, t)

    # Compute the next diffused state x_{t+1} using:
    # - the scaled version of x_t to simulate the forward process to the next step
    # - added noise scaled according to the noise level at t+1
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
        extract(coeff.sigmas, t+1, x_start.shape) * noise

    return x_t, x_t_plus_one
