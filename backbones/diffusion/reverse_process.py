# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
from typing import Tuple

import torch

from backbones.diffusion.utils import extract, get_sigma_schedule


class Posterior_Coefficients():
    """
    A class to compute and store coefficients necessary for the reverse diffusion process
    in Denoising Diffusion Probabilistic Models (DDPMs).
    """

    def __init__(self, n_timestep: int, beta_min: float, beta_max: float, device: torch.device, use_geometric: bool):
        """
        Initialize the Posterior_Coefficients class.

        Parameters
        ----------
        n_timestep : int
            The number of diffusion steps.
        beta_min : float
            The minimum beta value.
        beta_max : float
            The maximum beta value.
        device : torch.device
            The device to run the computations on.
        use_geometric : bool
            Whether to use geometric or variance preserving schedule.
        """

        _, _, self.betas = get_sigma_schedule(
            n_timestep=n_timestep, beta_min=beta_min, beta_max=beta_max, device=device, use_geometric=use_geometric)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)

        # Prepend 1 at the beginning for the initial state where no noise is added
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32,
             device=device), self.alphas_cumprod[:-1]), 0
        )

        # Variance of x_{t-1} given x_t:
        # Var(x_{t-1} | x_t, x_0) = beta_t * (1 - alpha_prod_{t-1}) / (1 - alpha_prod_t)
        self.posterior_variance = self.betas * \
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1 / self.alphas_cumprod - 1
        )

        # Coefficients for calculating the mean of x_{t-1} given x_t:
        # mu(x_{t-1} | x_t, x_0) = beta_t * sqrt(alpha_prod_{t-1}) / (1 - alpha_prod_t) * x_0
        #                   + (1 - alpha_prod_{t-1}) * sqrt(alpha_t) / (1 - alpha_prod_t) * x_t
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) /
            (1 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev) *
            torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)
        )

        # Clip the variance to avoid numerical instability
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients: Posterior_Coefficients, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Sample from the posterior distribution given x_0, x_t, and t.

    Parameters
    ----------
    coefficients : Posterior_Coefficients
        The coefficients for the reverse diffusion process.
    x_0 : torch.Tensor
        The initial state of the diffusion process, dimension [B, C, H, W].
    x_t : torch.Tensor
        The current state of the diffusion process, dimension [B, C, H, W].
    t : torch.Tensor
        The current timestep, dimension [B].

    Returns
    -------
    torch.Tensor
        The sampled state from the posterior distribution, dimension [B, C, H, W].
    """

    def q_posterior(x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the posterior mean, variance, and log variance given x_0, x_t, and t.

        Parameters
        ----------
        x_0 : torch.Tensor
            The initial state of the diffusion process, dimension [B, C, H, W].
        x_t : torch.Tensor
            The current state of the diffusion process, dimension [B, C, H, W].
        t : torch.Tensor
            The current timestep, dimension [B].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The posterior mean, variance, and log variance, dimension [B, C, H, W].
        """
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(
            coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from the posterior distribution given x_0, x_t, and t.

        Parameters
        ----------
        x_0 : torch.Tensor
            The initial state of the diffusion process, dimension [B, C, H, W].
        x_t : torch.Tensor
            The current state of the diffusion process, dimension [B, C, H, W].
        t : torch.Tensor
            The current timestep, dimension [B].

        Returns
        -------
        torch.Tensor
            The sampled state from the posterior distribution, dimension [B, C, H, W].
        """
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_from_model(coefficients: Posterior_Coefficients, generator: torch.nn.Module, n_time: int, x_init: torch.Tensor, T: int, latent_dimension: int) -> torch.Tensor:
    """
    Sample from the generative model given the initial state, diffusion process, and latent dimension.

    Parameters
    ----------
    coefficients : Posterior_Coefficients
        The coefficients for the reverse diffusion process.
    generator : torch.nn.Module
        The generative model.
    n_time : int
        The number of diffusion steps.
    x_init : torch.Tensor
        The initial state of the diffusion process, dimension [B, C, H, W].
    T : int
        The number of diffusion steps.
    latent_dimension : int
        The dimension of the latent space.

    Returns
    -------
    torch.Tensor
        The sampled state from the generative model, dimension [B, C, H, W].
    """
    x = x_init[:, [0], :]
    source = x_init[:, [1], :]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(
                x.size(0), latent_dimension, device=x.device)
            x_0 = generator(torch.cat((x, source), axis=1), t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0[:, [0], :], x, t)
            x = x_new.detach()

    return x
