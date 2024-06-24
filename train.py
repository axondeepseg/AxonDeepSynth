import os
from typing import Callable

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import backbones.generator_resnet
from backbones.diffusion.forward_process import (Diffusion_Coefficients,
                                                 q_sample_pairs)
from backbones.diffusion.reverse_process import (Posterior_Coefficients,
                                                 sample_from_model,
                                                 sample_posterior)
from backbones.diffusion.utils import broadcast_params, get_time_schedule
from backbones.discriminator import Discriminator_large
from backbones.ncsnpp_generator_adagn import NCSNpp
from configs.syndiff import SyndiffConfig
from dataset import CreateDatasetSynthesis
from utils.EMA import EMA
from utils.utils import to_range_0_1


def train_syndiff(
    global_rank: int,
    gpu: int,
    cfg: SyndiffConfig,
    world_size: int
):
    """
    Train the SynDiff model, a novel diffusion model for efficient, high-fidelity translation between source and target modalities of a given anatomy. 
    SynDiff incorporates a diffusive module with a source-conditional adversarial projector for fast and accurate reverse diffusion sampling, 
    and a non-diffusive module for unsupervised learning by estimating source images paired with corresponding target images.

    The training involves:
    1) Adversarial Diffusion Process: Utilizing a fast diffusion process with a conditional adversarial approach to model complex transition probabilities in reverse diffusion, enabling efficient image generation.
    2) Network Architecture: Employing a cycle-consistent architecture that leverages both diffusive and non-diffusive modules to learn from unpaired images, facilitating bilateral translation between two modalities.
    3) Learning Procedures: Implementing unsupervised learning through a cycle-consistency loss, comparing true target images against their reconstructions from both modules, and optimizing adversarial losses to train the model effectively without pretraining.

    This function initializes and orchestrates the training process, setting up the necessary configurations, data loaders, and training loops required to train the SynDiff model on the specified dataset.

    Parameters
    ----------
    global_rank: int
        The rank of the process.
    gpu: int
        The GPU to use.
    cfg: DictConfig
        Hydra configuration specifying various options and parameters for the SynDiff model.
    world_size: int
        Total number of processes.
    """

    # Set the seed for reproducibility
    torch.manual_seed(cfg.seed + global_rank)
    torch.cuda.manual_seed(cfg.seed + global_rank)
    torch.cuda.manual_seed_all(cfg.seed + global_rank)
    np.random.seed(cfg.seed + global_rank)

    # Set the device
    device = torch.device('cuda:{}'.format(gpu))

    # Get the datasets
    dataset_train = CreateDatasetSynthesis(
        phase="train",
        input_path=cfg.training_config.training_dataset_path,
        contrast1=cfg.contrast1,
        contrast2=cfg.contrast2,
        size=cfg.model_config.image_size,
        paired=cfg.training_config.paired
    )
    dataset_val = CreateDatasetSynthesis(
        phase="val",
        input_path=cfg.training_config.training_dataset_path,
        contrast1=cfg.contrast1,
        contrast2=cfg.contrast2,
        size=cfg.model_config.image_size,
        paired=cfg.training_config.paired
    )

    # Get the data loaders
    train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=global_rank)
    data_loader = DataLoader(
        dataset_train,
        batch_size=cfg.training_config.optimization_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    val_sampler = DistributedSampler(dataset_val, num_replicas=world_size, rank=global_rank)
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=cfg.training_config.optimization_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True
    )

    # Initialize the arrays to store the losses and metrics
    val_l1_loss = np.zeros([2, cfg.training_config.optimization_config.num_epoch+1, len(data_loader_val)])
    val_psnr_values = np.zeros([2, cfg.training_config.optimization_config.num_epoch+1, len(data_loader_val)])

    print('train data size:'+str(len(data_loader)))
    print('val data size:'+str(len(data_loader_val)))

    # networks performing reverse denoising
    gen_diffusive_1 = NCSNpp(
        image_size=cfg.model_config.image_size,
        num_channels=cfg.model_config.num_channels,
        num_channels_dae=cfg.model_config.num_channels_dae,
        ch_mult=cfg.model_config.ch_mult,
        num_res_blocks=cfg.model_config.num_res_blocks,
        attn_resolutions=cfg.model_config.attn_resolutions,
        dropout=cfg.training_config.optimization_config.dropout,
        resamp_with_conv=cfg.model_config.resamp_with_conv,
        conditional=cfg.model_config.conditional,
        fir=cfg.model_config.fir,
        fir_kernel=cfg.model_config.fir_kernel,
        skip_rescale=cfg.model_config.skip_rescale,
        resblock_type=cfg.model_config.resblock_type,
        progressive=cfg.model_config.progressive,
        progressive_input=cfg.model_config.progressive_input,
        embedding_type=cfg.model_config.embedding_type,
        fourier_scale=cfg.model_config.fourier_scale,
        not_use_tanh=cfg.model_config.not_use_tanh,
        z_emb_dim=cfg.model_config.z_emb_dim,
        progressive_combine=cfg.model_config.progressive_combine,
        n_mlp=cfg.model_config.n_mlp,
        latent_dim=cfg.model_config.latent_dim,
    ).to(device)
    gen_diffusive_2 = NCSNpp(
        image_size=cfg.model_config.image_size,
        num_channels=cfg.model_config.num_channels,
        num_channels_dae=cfg.model_config.num_channels_dae,
        ch_mult=cfg.model_config.ch_mult,
        num_res_blocks=cfg.model_config.num_res_blocks,
        attn_resolutions=cfg.model_config.attn_resolutions,
        dropout=cfg.training_config.optimization_config.dropout,
        resamp_with_conv=cfg.model_config.resamp_with_conv,
        conditional=cfg.model_config.conditional,
        fir=cfg.model_config.fir,
        fir_kernel=cfg.model_config.fir_kernel,
        skip_rescale=cfg.model_config.skip_rescale,
        resblock_type=cfg.model_config.resblock_type,
        progressive=cfg.model_config.progressive,
        progressive_input=cfg.model_config.progressive_input,
        embedding_type=cfg.model_config.embedding_type,
        fourier_scale=cfg.model_config.fourier_scale,
        not_use_tanh=cfg.model_config.not_use_tanh,
        z_emb_dim=cfg.model_config.z_emb_dim,
        progressive_combine=cfg.model_config.progressive_combine,
        n_mlp=cfg.model_config.n_mlp,
        latent_dim=cfg.model_config.latent_dim,
    ).to(device)

    # networks performing translation
    gen_non_diffusive_1to2 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[gpu])
    gen_non_diffusive_2to1 = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[gpu])

    # Define the discriminators used in the diffusion model
    disc_diffusive_1 = Discriminator_large(nc=2, ngf=cfg.model_config.ngf, t_emb_dim=cfg.model_config.t_emb_dim, act=nn.LeakyReLU(0.2)).to(device)
    disc_diffusive_2 = Discriminator_large(nc=2, ngf=cfg.model_config.ngf, t_emb_dim=cfg.model_config.t_emb_dim, act=nn.LeakyReLU(0.2)).to(device)

    # Define the discriminators used in the cycle-gan model
    disc_non_diffusive_cycle1 = backbones.generator_resnet.define_D(gpu_ids=[gpu])
    disc_non_diffusive_cycle2 = backbones.generator_resnet.define_D(gpu_ids=[gpu])

    # Broadcast the parameters to all GPUs
    broadcast_params(gen_diffusive_1.parameters())
    broadcast_params(gen_diffusive_2.parameters())
    broadcast_params(gen_non_diffusive_1to2.parameters())
    broadcast_params(gen_non_diffusive_2to1.parameters())
    broadcast_params(disc_diffusive_1.parameters())
    broadcast_params(disc_diffusive_2.parameters())
    broadcast_params(disc_non_diffusive_cycle1.parameters())
    broadcast_params(disc_non_diffusive_cycle2.parameters())

    # Initialize the optimizers
    optimizer_disc_diffusive_1 = optim.Adam(
        disc_diffusive_1.parameters(),
        lr=cfg.training_config.optimization_config.lr_d,
        betas=(cfg.training_config.optimization_config.beta1, cfg.training_config.optimization_config.beta2)
    )
    optimizer_disc_diffusive_2 = optim.Adam(
        disc_diffusive_2.parameters(),
        lr=cfg.training_config.optimization_config.lr_d,
        betas=(cfg.training_config.optimization_config.beta1, cfg.training_config.optimization_config.beta2)
    )

    optimizer_gen_diffusive_1 = optim.Adam(
        gen_diffusive_1.parameters(),
        lr=cfg.training_config.optimization_config.lr_g,
        betas=(cfg.training_config.optimization_config.beta1, cfg.training_config.optimization_config.beta2)
    )
    optimizer_gen_diffusive_2 = optim.Adam(
        gen_diffusive_2.parameters(),
        lr=cfg.training_config.optimization_config.lr_g,
        betas=(cfg.training_config.optimization_config.beta1, cfg.training_config.optimization_config.beta2)
    )

    optimizer_gen_non_diffusive_1to2 = optim.Adam(
        gen_non_diffusive_1to2.parameters(),
        lr=cfg.training_config.optimization_config.lr_g,
        betas=(cfg.training_config.optimization_config.beta1, cfg.training_config.optimization_config.beta2)
    )
    optimizer_gen_non_diffusive_2to1 = optim.Adam(
        gen_non_diffusive_2to1.parameters(),
        lr=cfg.training_config.optimization_config.lr_g,
        betas=(cfg.training_config.optimization_config.beta1, cfg.training_config.optimization_config.beta2)
    )

    optimizer_disc_non_diffusive_cycle1 = optim.Adam(
        disc_non_diffusive_cycle1.parameters(),
        lr=cfg.training_config.optimization_config.lr_d,
        betas=(cfg.training_config.optimization_config.beta1, cfg.training_config.optimization_config.beta2)
    )
    optimizer_disc_non_diffusive_cycle2 = optim.Adam(
        disc_non_diffusive_cycle2.parameters(),
        lr=cfg.training_config.optimization_config.lr_d,
        betas=(cfg.training_config.optimization_config.beta1, cfg.training_config.optimization_config.beta2)
    )

    # Initialize the exponential moving averages
    if cfg.training_config.optimization_config.use_ema:
        optimizer_gen_diffusive_1 = EMA(optimizer_gen_diffusive_1, ema_decay=cfg.training_config.optimization_config.ema_decay)
        optimizer_gen_diffusive_2 = EMA(optimizer_gen_diffusive_2, ema_decay=cfg.training_config.optimization_config.ema_decay)
        optimizer_gen_non_diffusive_1to2 = EMA(optimizer_gen_non_diffusive_1to2, ema_decay=cfg.training_config.optimization_config.ema_decay)
        optimizer_gen_non_diffusive_2to1 = EMA(optimizer_gen_non_diffusive_2to1, ema_decay=cfg.training_config.optimization_config.ema_decay)

    # Initialize the learning rate schedulers
    scheduler_gen_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_1, cfg.training_config.optimization_config.num_epoch, eta_min=1e-5)
    scheduler_gen_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_2, cfg.training_config.optimization_config.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_1to2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_non_diffusive_1to2, cfg.training_config.optimization_config.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_2to1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_non_diffusive_2to1, cfg.training_config.optimization_config.num_epoch, eta_min=1e-5)
    scheduler_disc_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_1, cfg.training_config.optimization_config.num_epoch, eta_min=1e-5)
    scheduler_disc_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_2, cfg.training_config.optimization_config.num_epoch, eta_min=1e-5)
    scheduler_disc_non_diffusive_cycle1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_non_diffusive_cycle1, cfg.training_config.optimization_config.num_epoch, eta_min=1e-5)
    scheduler_disc_non_diffusive_cycle2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_non_diffusive_cycle2, cfg.training_config.optimization_config.num_epoch, eta_min=1e-5)

    # ddp
    gen_diffusive_1 = nn.parallel.DistributedDataParallel(gen_diffusive_1, device_ids=[gpu])
    gen_diffusive_2 = nn.parallel.DistributedDataParallel(gen_diffusive_2, device_ids=[gpu])
    gen_non_diffusive_1to2 = nn.parallel.DistributedDataParallel(gen_non_diffusive_1to2, device_ids=[gpu])
    gen_non_diffusive_2to1 = nn.parallel.DistributedDataParallel(gen_non_diffusive_2to1, device_ids=[gpu])
    disc_diffusive_1 = nn.parallel.DistributedDataParallel(disc_diffusive_1, device_ids=[gpu])
    disc_diffusive_2 = nn.parallel.DistributedDataParallel(disc_diffusive_2, device_ids=[gpu])
    disc_non_diffusive_cycle1 = nn.parallel.DistributedDataParallel(disc_non_diffusive_cycle1, device_ids=[gpu])
    disc_non_diffusive_cycle2 = nn.parallel.DistributedDataParallel(disc_non_diffusive_cycle2, device_ids=[gpu])

    exp_path = os.path.join(cfg.syndiff_results_path, cfg.exp)
    if global_rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

    coeff = Diffusion_Coefficients(
        n_timestep=cfg.model_config.num_timesteps,
        beta_min=cfg.model_config.beta_min,
        beta_max=cfg.model_config.beta_max,
        device=device,
        use_geometric=cfg.model_config.use_geometric,
    )
    pos_coeff = Posterior_Coefficients(
        n_timestep=cfg.model_config.num_timesteps,
        beta_min=cfg.model_config.beta_min,
        beta_max=cfg.model_config.beta_max,
        device=device,
        use_geometric=cfg.model_config.use_geometric,
    )
    T = get_time_schedule(cfg.model_config.num_timesteps, device)

    if cfg.training_config.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch

        # Load the parameters of the generators from the checkpoint
        gen_diffusive_1.load_state_dict(checkpoint['gen_diffusive_1_dict'])
        gen_diffusive_2.load_state_dict(checkpoint['gen_diffusive_2_dict'])
        gen_non_diffusive_1to2.load_state_dict(checkpoint['gen_non_diffusive_1to2_dict'])
        gen_non_diffusive_2to1.load_state_dict(checkpoint['gen_non_diffusive_2to1_dict'])
        optimizer_gen_diffusive_1.load_state_dict(checkpoint['optimizer_gen_diffusive_1'])
        scheduler_gen_diffusive_1.load_state_dict(checkpoint['scheduler_gen_diffusive_1'])
        optimizer_gen_diffusive_2.load_state_dict(checkpoint['optimizer_gen_diffusive_2'])
        scheduler_gen_diffusive_2.load_state_dict(checkpoint['scheduler_gen_diffusive_2'])
        optimizer_gen_non_diffusive_1to2.load_state_dict(checkpoint['optimizer_gen_non_diffusive_1to2'])
        scheduler_gen_non_diffusive_1to2.load_state_dict(checkpoint['scheduler_gen_non_diffusive_1to2'])
        optimizer_gen_non_diffusive_2to1.load_state_dict(checkpoint['optimizer_gen_non_diffusive_2to1'])
        scheduler_gen_non_diffusive_2to1.load_state_dict(checkpoint['scheduler_gen_non_diffusive_2to1'])

        # Load the parameters of the discriminators
        disc_diffusive_1.load_state_dict(checkpoint['disc_diffusive_1_dict'])
        optimizer_disc_diffusive_1.load_state_dict(checkpoint['optimizer_disc_diffusive_1'])
        scheduler_disc_diffusive_1.load_state_dict(checkpoint['scheduler_disc_diffusive_1'])
        disc_diffusive_2.load_state_dict(checkpoint['disc_diffusive_2_dict'])
        optimizer_disc_diffusive_2.load_state_dict(checkpoint['optimizer_disc_diffusive_2'])
        scheduler_disc_diffusive_2.load_state_dict(checkpoint['scheduler_disc_diffusive_2'])
        disc_non_diffusive_cycle1.load_state_dict(checkpoint['disc_non_diffusive_cycle1_dict'])
        optimizer_disc_non_diffusive_cycle1.load_state_dict(checkpoint['optimizer_disc_non_diffusive_cycle1'])
        scheduler_disc_non_diffusive_cycle1.load_state_dict(checkpoint['scheduler_disc_non_diffusive_cycle1'])
        disc_non_diffusive_cycle2.load_state_dict(checkpoint['disc_non_diffusive_cycle2_dict'])
        optimizer_disc_non_diffusive_cycle2.load_state_dict(checkpoint['optimizer_disc_non_diffusive_cycle2'])
        scheduler_disc_non_diffusive_cycle2.load_state_dict(checkpoint['scheduler_disc_non_diffusive_cycle2'])

        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    for epoch in range(init_epoch, cfg.training_config.optimization_config.num_epoch+1):
        dataset_train.on_epoch_start()
        train_sampler.set_epoch(epoch)

        for iteration, (x1, x2) in enumerate(data_loader):
            for p in disc_diffusive_1.parameters():
                p.requires_grad = True
            for p in disc_diffusive_2.parameters():
                p.requires_grad = True
            for p in disc_non_diffusive_cycle1.parameters():
                p.requires_grad = True
            for p in disc_non_diffusive_cycle2.parameters():
                p.requires_grad = True

            # ----------------------------------------- Diffusive Step  Discriminator -----------------------------------------

            # Initialize the gradients to zero
            disc_diffusive_1.zero_grad()
            disc_diffusive_2.zero_grad()

            # sample from p(x_0)
            real_data1 = x1.to(device, non_blocking=True)
            real_data2 = x2.to(device, non_blocking=True)

            # sample t
            t1 = torch.randint(0, cfg.model_config.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, cfg.model_config.num_timesteps, (real_data2.size(0),), device=device)

            # sample x_t and x_{t+1}
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            x1_t.requires_grad = True
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)
            x2_t.requires_grad = True

            # train discriminator with real
            D1_real = disc_diffusive_1(x1_t, t1, x1_tp1.detach()).view(-1)
            D2_real = disc_diffusive_2(x2_t, t2, x2_tp1.detach()).view(-1)

            # Calculate the diffusive discriminator loss for real data
            errD1_real = F.softplus(-D1_real)
            errD1_real = errD1_real.mean()
            errD2_real = F.softplus(-D2_real)
            errD2_real = errD2_real.mean()
            errD_real = errD1_real + errD2_real
            errD_real.backward(retain_graph=True)

            if not cfg.training_config.optimization_config.lazy_reg:
                grad1_real = torch.autograd.grad(outputs=D1_real.sum(), inputs=x1_t, create_graph=True)[0]
                grad1_penalty = (grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad2_real = torch.autograd.grad(outputs=D2_real.sum(), inputs=x2_t, create_graph=True)[0]
                grad2_penalty = (grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2).mean()

                grad_penalty = cfg.training_config.optimization_config.r1_gamma / 2 * grad1_penalty + cfg.training_config.optimization_config.r1_gamma / 2 * grad2_penalty
                grad_penalty.backward()
            else:
                if global_step % cfg.training_config.optimization_config.lazy_reg == 0:
                    grad1_real = torch.autograd.grad(outputs=D1_real.sum(), inputs=x1_t, create_graph=True)[0]
                    grad1_penalty = (grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad2_real = torch.autograd.grad(outputs=D2_real.sum(), inputs=x2_t, create_graph=True)[0]
                    grad2_penalty = (grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2).mean()

                    grad_penalty = cfg.training_config.optimization_config.r1_gamma / 2 * grad1_penalty + cfg.training_config.optimization_config.r1_gamma / 2 * grad2_penalty
                    grad_penalty.backward()

            # train with fake
            latent_z1 = torch.randn(cfg.training_config.optimization_config.batch_size, cfg.model_config.latent_dim, device=device)
            latent_z2 = torch.randn(cfg.training_config.optimization_config.batch_size, cfg.model_config.latent_dim, device=device)

            # Generate the predicted x_0 for the diffusive model with cycle-gan generators
            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)

            # x_tp1 is concatenated with source contrast and x_0_predict is predicted
            x1_0_predict_diff = gen_diffusive_1(torch.cat((x1_tp1.detach(), x2_0_predict), axis=1), t1, latent_z1)
            x2_0_predict_diff = gen_diffusive_2(torch.cat((x2_tp1.detach(), x1_0_predict), axis=1), t2, latent_z2)

            # sampling q(x_t | x_0_predict, x_t+1)
            x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:, [0], :], x1_tp1, t1)
            x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:, [0], :], x2_tp1, t2)

            # D output for fake sample x_pos_sample
            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
            output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)

            # Calculate the diffusive discriminator loss for fake data
            errD1_fake = F.softplus(output1)
            errD2_fake = F.softplus(output2)
            errD_fake = errD1_fake.mean() + errD2_fake.mean()
            errD_fake.backward()

            # Calculate the total discriminator loss
            errD = errD_real + errD_fake

            # Update D
            optimizer_disc_diffusive_1.step()
            optimizer_disc_diffusive_2.step()

            # ----------------------------------------- Non-Diffusive Step Discriminator -----------------------------------------

            # Initialize the gradients to zero
            disc_non_diffusive_cycle1.zero_grad()
            disc_non_diffusive_cycle2.zero_grad()

            # sample from p(x_0)
            real_data1 = x1.to(device, non_blocking=True)
            real_data2 = x2.to(device, non_blocking=True)

            D_cycle1_real = disc_non_diffusive_cycle1(real_data1).view(-1)
            D_cycle2_real = disc_non_diffusive_cycle2(real_data2).view(-1)

            errD_cycle1_real = F.softplus(-D_cycle1_real)
            errD_cycle1_real = errD_cycle1_real.mean()

            errD_cycle2_real = F.softplus(-D_cycle2_real)
            errD_cycle2_real = errD_cycle2_real.mean()
            errD_cycle_real = errD_cycle1_real + errD_cycle2_real
            errD_cycle_real.backward(retain_graph=True)

            # train with fake
            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)

            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1)

            errD_cycle1_fake = F.softplus(D_cycle1_fake)
            errD_cycle1_fake = errD_cycle1_fake.mean()

            errD_cycle2_fake = F.softplus(D_cycle2_fake)
            errD_cycle2_fake = errD_cycle2_fake.mean()
            errD_cycle_fake = errD_cycle1_fake + errD_cycle2_fake
            errD_cycle_fake.backward()

            # Calculate the total non-diffusive discriminator loss
            errD_cycle = errD_cycle_real + errD_cycle_fake

            # Update D
            optimizer_disc_non_diffusive_cycle1.step()
            optimizer_disc_non_diffusive_cycle2.step()

            # ----------------------------------------- Generator -----------------------------------------

            # Freeze the discriminators
            for p in disc_diffusive_1.parameters():
                p.requires_grad = False
            for p in disc_diffusive_2.parameters():
                p.requires_grad = False
            for p in disc_non_diffusive_cycle1.parameters():
                p.requires_grad = False
            for p in disc_non_diffusive_cycle2.parameters():
                p.requires_grad = False
            gen_diffusive_1.zero_grad()
            gen_diffusive_2.zero_grad()
            gen_non_diffusive_1to2.zero_grad()
            gen_non_diffusive_2to1.zero_grad()

            t1 = torch.randint(0, cfg.model_config.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, cfg.model_config.num_timesteps, (real_data2.size(0),), device=device)

            # sample x_t and x_tp1
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)

            latent_z1 = torch.randn(cfg.training_config.optimization_config.batch_size, cfg.model_config.latent_dim, device=device)
            latent_z2 = torch.randn(cfg.training_config.optimization_config.batch_size, cfg.model_config.latent_dim, device=device)

            # translation networks
            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict_cycle = gen_non_diffusive_1to2(x1_0_predict)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)
            x1_0_predict_cycle = gen_non_diffusive_2to1(x2_0_predict)

            # x_tp1 is concatenated with source contrast and x_0_predict is predicted
            x1_0_predict_diff = gen_diffusive_1(torch.cat((x1_tp1.detach(), x2_0_predict), axis=1), t1, latent_z1)
            x2_0_predict_diff = gen_diffusive_2(torch.cat((x2_tp1.detach(), x1_0_predict), axis=1), t2, latent_z2)
            # sampling q(x_t | x_0_predict, x_t+1)
            x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:, [0], :], x1_tp1, t1)
            x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:, [0], :], x2_tp1, t2)
            # D output for fake sample x_pos_sample
            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
            output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)

            errG1 = F.softplus(-output1)
            errG1 = errG1.mean()

            errG2 = F.softplus(-output2)
            errG2 = errG2.mean()

            errG_adv = errG1 + errG2

            # D_cycle output for fake x1_0_predict
            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1)

            errG_cycle_adv1 = F.softplus(-D_cycle1_fake)
            errG_cycle_adv1 = errG_cycle_adv1.mean()

            errG_cycle_adv2 = F.softplus(-D_cycle2_fake)
            errG_cycle_adv2 = errG_cycle_adv2.mean()
            errG_cycle_adv = errG_cycle_adv1 + errG_cycle_adv2

            # L1 loss
            errG1_L1 = F.l1_loss(x1_0_predict_diff[:, [0], :], real_data1)
            errG2_L1 = F.l1_loss(x2_0_predict_diff[:, [0], :], real_data2)
            errG_L1 = errG1_L1 + errG2_L1

            # cycle loss
            errG1_cycle = F.l1_loss(x1_0_predict_cycle, real_data1)
            errG2_cycle = F.l1_loss(x2_0_predict_cycle, real_data2)
            errG_cycle = errG1_cycle + errG2_cycle

            torch.autograd.set_detect_anomaly(True)

            errG = cfg.training_config.optimization_config.lambda_l1_loss*errG_cycle + errG_adv + \
                errG_cycle_adv + cfg.training_config.optimization_config.lambda_l1_loss*errG_L1
            errG.backward()

            optimizer_gen_diffusive_1.step()
            optimizer_gen_diffusive_2.step()
            optimizer_gen_non_diffusive_1to2.step()
            optimizer_gen_non_diffusive_2to1.step()

            global_step += 1
            if iteration % 100 == 0:
                if global_rank == 0:
                    print('epoch {} iteration{}, G-Cycle: {}, G-L1: {}, G-Adv: {}, G-cycle-Adv: {}, G-Sum: {}, D Loss: {}, D_cycle Loss: {}'.format(epoch,
                          iteration, errG_cycle.item(), errG_L1.item(),  errG_adv.item(), errG_cycle_adv.item(), errG.item(), errD.item(), errD_cycle.item()))

        if not cfg.training_config.optimization_config.no_lr_decay:

            scheduler_gen_diffusive_1.step()
            scheduler_gen_diffusive_2.step()
            scheduler_gen_non_diffusive_1to2.step()
            scheduler_gen_non_diffusive_2to1.step()
            scheduler_disc_diffusive_1.step()
            scheduler_disc_diffusive_2.step()

            scheduler_disc_non_diffusive_cycle1.step()
            scheduler_disc_non_diffusive_cycle2.step()

        if global_rank == 0:
            if epoch % 10 == 0:
                torchvision.utils.save_image(x1_pos_sample, os.path.join(exp_path, 'xpos1_epoch_{}.png'.format(epoch)), normalize=True)
                torchvision.utils.save_image(x2_pos_sample, os.path.join(exp_path, 'xpos2_epoch_{}.png'.format(epoch)), normalize=True)
            # concatenate noise and source contrast
            x1_t = torch.cat((torch.randn_like(real_data1), real_data2), axis=1)
            fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, cfg.model_config.num_timesteps, x1_t, T, latent_dimension=cfg.model_config.latent_dim)
            fake_sample1 = torch.cat((real_data2, fake_sample1), axis=-1)
            torchvision.utils.save_image(fake_sample1, os.path.join(exp_path, 'sample1_discrete_epoch_{}.png'.format(epoch)), normalize=True)
            pred1 = gen_non_diffusive_2to1(real_data2)
            #
            x2_t = torch.cat((torch.randn_like(real_data2), pred1), axis=1)
            fake_sample2_tilda = gen_diffusive_2(x2_t, t2, latent_z2)
            #
            pred1 = torch.cat((real_data2, pred1, gen_non_diffusive_1to2(pred1), fake_sample2_tilda[:, [0], :]), axis=-1)
            torchvision.utils.save_image(pred1, os.path.join(exp_path, 'sample1_translated_epoch_{}.png'.format(epoch)), normalize=True)

            x2_t = torch.cat((torch.randn_like(real_data2), real_data1), axis=1)
            fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, cfg.model_config.num_timesteps, x2_t, T, latent_dimension=cfg.model_config.latent_dim)
            fake_sample2 = torch.cat((real_data1, fake_sample2), axis=-1)
            torchvision.utils.save_image(fake_sample2, os.path.join(exp_path, 'sample2_discrete_epoch_{}.png'.format(epoch)), normalize=True)
            pred2 = gen_non_diffusive_1to2(real_data1)
            #
            x1_t = torch.cat((torch.randn_like(real_data1), pred2), axis=1)
            fake_sample1_tilda = gen_diffusive_1(x1_t, t1, latent_z1)
            #
            pred2 = torch.cat((real_data1, pred2, gen_non_diffusive_2to1(pred2), fake_sample1_tilda[:, [0], :]), axis=-1)
            torchvision.utils.save_image(pred2, os.path.join(exp_path, 'sample2_translated_epoch_{}.png'.format(epoch)), normalize=True)

            if cfg.training_config.save_content:
                if epoch % cfg.training_config.save_content_every == 0:
                    print('Saving content.')
                    content = {
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'args': dict(cfg),
                        'gen_diffusive_1_dict': gen_diffusive_1.state_dict(),
                        'optimizer_gen_diffusive_1': optimizer_gen_diffusive_1.state_dict(),
                        'gen_diffusive_2_dict': gen_diffusive_2.state_dict(),
                        'optimizer_gen_diffusive_2': optimizer_gen_diffusive_2.state_dict(),
                        'scheduler_gen_diffusive_1': scheduler_gen_diffusive_1.state_dict(),
                        'disc_diffusive_1_dict': disc_diffusive_1.state_dict(),
                        'scheduler_gen_diffusive_2': scheduler_gen_diffusive_2.state_dict(),
                        'disc_diffusive_2_dict': disc_diffusive_2.state_dict(),
                        'gen_non_diffusive_1to2_dict': gen_non_diffusive_1to2.state_dict(),
                        'optimizer_gen_non_diffusive_1to2': optimizer_gen_non_diffusive_1to2.state_dict(),
                        'gen_non_diffusive_2to1_dict': gen_non_diffusive_2to1.state_dict(),
                        'optimizer_gen_non_diffusive_2to1': optimizer_gen_non_diffusive_2to1.state_dict(),
                        'scheduler_gen_non_diffusive_1to2': scheduler_gen_non_diffusive_1to2.state_dict(),
                        'scheduler_gen_non_diffusive_2to1': scheduler_gen_non_diffusive_2to1.state_dict(),
                        'optimizer_disc_diffusive_1': optimizer_disc_diffusive_1.state_dict(),
                        'scheduler_disc_diffusive_1': scheduler_disc_diffusive_1.state_dict(),
                        'optimizer_disc_diffusive_2': optimizer_disc_diffusive_2.state_dict(),
                        'scheduler_disc_diffusive_2': scheduler_disc_diffusive_2.state_dict(),
                        'optimizer_disc_non_diffusive_cycle1': optimizer_disc_non_diffusive_cycle1.state_dict(),
                        'scheduler_disc_non_diffusive_cycle1': scheduler_disc_non_diffusive_cycle1.state_dict(),
                        'optimizer_disc_non_diffusive_cycle2': optimizer_disc_non_diffusive_cycle2.state_dict(),
                        'scheduler_disc_non_diffusive_cycle2': scheduler_disc_non_diffusive_cycle2.state_dict(),
                        'disc_non_diffusive_cycle1_dict': disc_non_diffusive_cycle1.state_dict(),
                        'disc_non_diffusive_cycle2_dict': disc_non_diffusive_cycle2.state_dict()}

                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % cfg.training_config.save_ckpt_every == 0:
                if cfg.training_config.optimization_config.use_ema:
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_1to2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_2to1.swap_parameters_with_ema(store_params_in_ema=True)
                torch.save(gen_diffusive_1.state_dict(), os.path.join(exp_path, 'gen_diffusive_1_{}.pth'.format(epoch)))
                torch.save(gen_diffusive_2.state_dict(), os.path.join(exp_path, 'gen_diffusive_2_{}.pth'.format(epoch)))
                torch.save(gen_non_diffusive_1to2.state_dict(), os.path.join(exp_path, 'gen_non_diffusive_1to2_{}.pth'.format(epoch)))
                torch.save(gen_non_diffusive_2to1.state_dict(), os.path.join(exp_path, 'gen_non_diffusive_2to1_{}.pth'.format(epoch)))
                if cfg.training_config.optimization_config.use_ema:
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_1to2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_2to1.swap_parameters_with_ema(store_params_in_ema=True)

        for iteration, (x_val, y_val) in enumerate(data_loader_val):

            real_data = x_val.to(device, non_blocking=True)
            source_data = y_val.to(device, non_blocking=True)

            x1_t = torch.cat((torch.randn_like(real_data), source_data), axis=1)
            # diffusion steps
            fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, cfg.model_config.num_timesteps, x1_t, T, latent_dimension=cfg.model_config.latent_dim)
            fake_sample1 = to_range_0_1(fake_sample1)
            fake_sample1 = fake_sample1/fake_sample1.mean()
            real_data = to_range_0_1(real_data)
            real_data = real_data/real_data.mean()

            fake_sample1 = fake_sample1.cpu().numpy()
            real_data = real_data.cpu().numpy()
            val_l1_loss[0, epoch, iteration] = abs(fake_sample1 - real_data).mean()

            val_psnr_values[0, epoch, iteration] = psnr(real_data, fake_sample1, data_range=real_data.max())

        print(np.nanmean(val_psnr_values[0, epoch, :]))
        print(np.nanmean(val_psnr_values[1, epoch, :]))
        np.save('{}/val_l1_loss.npy'.format(exp_path), val_l1_loss)
        np.save('{}/val_psnr_values.npy'.format(exp_path), val_psnr_values)


def init_processes(global_rank: int, fn: Callable, cfg: SyndiffConfig, world_size: int, gpu: int):
    """ 
    Initialize the distributed environment.

    Parameters
    ----------
    global_rank : int
        Rank of the process.
    fn : function
        Function to be executed by the process.
    cfg : SyndiffConfig
        Configuration for the training process.
    world_size : int
        Total number of processes across all nodes.
    gpu : int
        GPU device to be used by the process.
    """
    # Setup the distributed environment
    os.environ['MASTER_ADDR'] = cfg.network_distribution.master_address
    os.environ['MASTER_PORT'] = cfg.network_distribution.port_num

    # Set the GPU device
    torch.cuda.set_device(gpu)

    # Initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=global_rank,
        world_size=world_size
    )

    # Unpack all arguments from args and pass them individually to fn
    fn(global_rank=global_rank, gpu=gpu, cfg=cfg, world_size=world_size)

    dist.barrier()
    cleanup()


def cleanup():
    """
    Clean up the distributed process group.

    This function is called to destroy the process group once all processes have completed their tasks.
    It ensures that all resources allocated for the distributed processes are properly released.
    """
    dist.destroy_process_group()


def worker(local_rank, cfg: SyndiffConfig, world_size: int):
    """
    Initialize the distributed environment and train the syndiff model using the specified configuration.

    Parameters
    ----------
    rank : int
        Rank of the process.
    cfg : SyndiffConfig
        Configuration for the training process.
    world_size : int
        Total number of processes across all nodes.
    """
    global_rank = local_rank + cfg.network_distribution.node_rank * len(cfg.network_distribution.gpus)
    init_processes(global_rank, train_syndiff, cfg, world_size, gpu=cfg.network_distribution.gpus[local_rank])


@hydra.main(config_path="configs", config_name="syndiff")
def main(cfg: SyndiffConfig):
    """
    Main function to train the syndiff model.

    This function initializes the distributed environment and trains the syndiff model using the specified configuration.
    It supports multi-GPU training and supports data parallelism.

    Parameters
    ----------
    cfg : SyndiffConfig
        Configuration for the training process.
    """
    OmegaConf.set_struct(cfg, False)  # Allow adding new keys

    # Set the number of processes per node
    num_process_per_node = len(cfg.network_distribution.gpus)

    # Set the world size, which is the number of processes across all nodes
    world_size = cfg.network_distribution.num_proc_node * num_process_per_node

    if num_process_per_node > 1:
        mp.spawn(worker, nprocs=num_process_per_node, args=(cfg, world_size))
    else:
        worker(0, cfg, world_size)


if __name__ == "__main__":
    main()
