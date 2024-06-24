import json
import os
from collections import defaultdict

import hydra
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbones.diffusion.reverse_process import Posterior_Coefficients
from backbones.diffusion.utils import get_time_schedule, load_checkpoint
from backbones.ncsnpp_generator_adagn import NCSNpp
from configs.syndiff import SyndiffConfig
from dataset import CreateDatasetSynthesis
from evaluation.evaluate import Evaluator


@hydra.main(config_path="./configs", config_name="syndiff.yaml")
def test(cfg: SyndiffConfig):
    """
    Translates a dataset from one imaging modality to another using a trained SynDiff model and
    creates a dataset in nnU-Net format for the translated images.

    This function takes a dataset in a specific imaging modality (source) and translates it into
    another modality (target), using a pre-trained SynDiff generative model. The translated dataset
    is then organized into the nnU-Net format, which involves creating directories for training
    images, training labels, and test images under the nnU-Net raw data directory. This enables
    the use of the translated dataset for further medical image segmentation tasks using nnU-Net.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration specifying various options and parameters for the
        translation process. Important arguments include the source and target modalities,
        dataset paths, model checkpoint paths, device configuration, and nnU-Net directory
        paths.

    Raises
    ------
    AssertionError
        If the nnU-Net directory specified in the arguments does not exist.
    """

    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)

    # Set device
    torch.cuda.set_device(cfg.network_distribution.gpus[0])
    device = torch.device('cuda:{}'.format(cfg.network_distribution.gpus[0]) if len(cfg.network_distribution.gpus) > 0 else 'cpu')

    # Set epoch from which to load the model
    epoch_chosen = cfg.translation_config.which_epoch

    # Get the datasets
    dataset_test = CreateDatasetSynthesis(
        phase="test",
        input_path=cfg.training_config.training_dataset_path,
        contrast1=cfg.contrast1,
        contrast2=cfg.contrast2,
        size=cfg.model_config.image_size,
        paired=cfg.training_config.paired
    )

    # Get the data loaders
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=cfg.training_config.optimization_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    print('test data size:'+str(len(data_loader_test)))

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

    # Define the path to the checkpoint file
    exp_path = os.path.join(cfg.syndiff_results_path, cfg.exp)
    checkpoint_file = exp_path + "/{}_{}.pth"

    # Load the checkpoint
    load_checkpoint(
        checkpoint_file,
        gen_diffusive_1,
        "gen_diffusive_1",
        epoch=str(epoch_chosen),
        device=device,
    )

    load_checkpoint(
        checkpoint_file,
        gen_diffusive_2,
        "gen_diffusive_2",
        epoch=str(epoch_chosen),
        device=device,
    )

    # Get the time schedule for diffusion
    T = get_time_schedule(cfg.model_config.num_timesteps, device)

    # Initialize the posterior coefficients used for sampling from the diffusion model
    pos_coeff = Posterior_Coefficients(
        n_timestep=cfg.model_config.num_timesteps,
        beta_min=cfg.model_config.beta_min,
        beta_max=cfg.model_config.beta_max,
        device=device,
        use_geometric=cfg.model_config.use_geometric,
    )

    results = defaultdict(lambda: defaultdict(list))
    evaluator = Evaluator(
        num_timesteps=cfg.model_config.num_timesteps,
        time_schedule=T,
        latent_dim=cfg.model_config.latent_dim,
        device=device,
        paired=cfg.training_config.paired
    )
    for x_test, y_test in tqdm(data_loader_test, desc="Testing"):
        results = evaluator.update(
            results,
            x_test.to(device),
            y_test.to(device),
            gen_diffusive_1,
            gen_diffusive_2,
            pos_coeff
        )

    # Save the results to a JSON file
    results_file_path = os.path.join(exp_path, "evaluation_results_epoch{}.json".format(epoch_chosen))
    with open(results_file_path, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    test()
