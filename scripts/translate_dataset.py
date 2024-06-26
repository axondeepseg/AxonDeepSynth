import json
import os
import re
import shutil
import tempfile
from collections import defaultdict

import cv2
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbones.diffusion.reverse_process import (Posterior_Coefficients,
                                                 sample_from_model)
from backbones.diffusion.utils import get_time_schedule, load_checkpoint
from backbones.ncsnpp_generator_adagn import NCSNpp
from configs.syndiff import SyndiffConfig
from dataset import DatasetToBeTranslated
from utils.images import reconstruct_image_from_patches


def setup_nnunet_dataset(cfg: SyndiffConfig, target_contrast: str, source_contrast: str, len_dataset: int):
    """
    Setups the nnU-Net dataset for the translated images.

    Parameters
    ----------
    cfg : SyndiffConfig
        The configuration of the SynDiff model.
    target_contrast : str
        The target contrast to be translated.
    source_contrast : str
        The source contrast to be translated.
    len_dataset : int
        The number of images in the dataset.

    Returns
    -------
    train_dir : str
        The path to the training images.
    train_labels_dir : str
        The path to the training labels.
    test_dir : str
        The path to the test images.
    """
    # Define the name of the dataset in nnU-Net format
    dataset_name = f"Dataset{cfg.segmentation_config.nnunet_dataset_id:03d}_SYNTH_{target_contrast}_from_{source_contrast}"

    # Ensure the nnU-Net directory exists
    assert os.path.exists(cfg.segmentation_config.nnunet_dir), "nnunet directory does not exist"

    # Define the paths to the training images, training labels, and test images
    train_dir = os.path.join(cfg.segmentation_config.nnunet_dir, "nnUNet_raw", dataset_name, "imagesTr")
    train_labels_dir = os.path.join(
        cfg.segmentation_config.nnunet_dir, "nnUNet_raw", dataset_name, "labelsTr"
    )
    test_dir = os.path.join(cfg.segmentation_config.nnunet_dir, "nnUNet_raw", dataset_name, "imagesTs")
    os.makedirs(
        train_dir,
        exist_ok=True,
    )
    os.makedirs(
        train_labels_dir,
        exist_ok=True,
    )
    os.makedirs(
        test_dir,
        exist_ok=True,
    )

    # Define the dataset JSON file used by nnU-Net
    dataset_json = {
        "name": f"SYNTH_{target_contrast}",
        "description": f"Synthetic {target_contrast} axon and myelin segmentation dataset for nnUNetv2",
        "labels": {"background": 0, "myelin": 1, "axon": 2},
        "channel_names": {"0": "rescale_to_0_1"},
        "numTraining": len_dataset,
        "numTest": 0,
        "file_ending": ".png",
    }

    # Save the dataset JSON file
    dataset_json_path = os.path.join(
        cfg.segmentation_config.nnunet_dir, "nnUNet_raw", dataset_name, "dataset.json"
    )
    with open(dataset_json_path, "w") as json_file:
        json.dump(dataset_json, json_file, indent=4)

    return train_dir, train_labels_dir, test_dir


@hydra.main(config_path="../configs", config_name="syndiff.yaml")
def translate_dataset(cfg: SyndiffConfig):
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

    # Define a function to rescale images to the range 0-255
    def to_range_0_255(x): return (x + 1.0) * 127.5

    # Loading dataset
    dataset = DatasetToBeTranslated(
        cfg.translation_config.path_dataset_to_translate,
        modality=cfg.translation_config.source_contrast,
        has_labels=cfg.translation_config.is_nnunet_dir,
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Initializing and loading network
    gen_diffusive = NCSNpp(
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
        gen_diffusive,
        "gen_diffusive_1" if cfg.translation_config.source_contrast == cfg.contrast2 else "gen_diffusive_2",
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

    # Determine the target contrast based on the source and contrast arguments
    target_contrast = (
        cfg.contrast1 if cfg.translation_config.source_contrast == cfg.contrast2 else cfg.contrast2
    )

    # Create temporary directories for storing images and labels
    image_dir_temp = tempfile.mkdtemp(prefix="image_dir_")
    labels_dir_temp = tempfile.mkdtemp(prefix="labels_dir_")

    try:
        image_to_patches = defaultdict(list)
        # Translate each image in the dataset and save the translated images and labels in the new nnU-Net formatted dataset
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Translating images"):
            image = batch["image"]
            if cfg.translation_config.is_nnunet_dir:
                label = batch["label"]

            # Move the image to the device
            source_data = image.to(device, non_blocking=True)

            # Concatenate a random noise vector and the source data to form the input for the diffusion model
            x_t = torch.cat((torch.randn_like(source_data), source_data), axis=1)

            # Perform diffusion steps to generate a fake sample from the diffusion model
            fake_sample = sample_from_model(
                pos_coeff, gen_diffusive, cfg.model_config.num_timesteps, x_t, T, cfg.model_config.latent_dim
            )

            # Rescale the fake sample to the range 0-255 and convert to a numpy array
            fake_sample = to_range_0_255(fake_sample).clamp(0, 255).cpu().numpy().astype(np.uint8)

            # Save the fake sample and label to the nnU-Net formatted dataset
            patch_dir = f"{image_dir_temp}/SYNTH_{target_contrast}_{i:03d}_0000.png"
            cv2.imwrite(
                patch_dir,
                fake_sample[0, 0],
            )

            image_data = {
                "patch_dir": patch_dir,
                "image_location": batch["image_location"].flatten().tolist(),
            }

            if cfg.translation_config.is_nnunet_dir:
                patch_label_dir = f"{labels_dir_temp}/SYNTH_{target_contrast}_{i:03d}.png"
                cv2.imwrite(
                    patch_label_dir,
                    label[0, 0].numpy(),
                )

                image_data["patch_label_dir"] = patch_label_dir
                image_data["label_location"] = batch["label_location"].flatten().tolist()
                image_data["label_path_original"] = batch["label_path_original"][0]

            image_to_patches[batch["image_path_original"][0]].append(image_data)

        metadata = {
            "patch_dimension": list(image.shape)[2:],
            "image_to_patches": image_to_patches,
        }

        if cfg.translation_config.is_nnunet_dir:
            image_dir, labels_dir, _ = setup_nnunet_dataset(cfg, target_contrast, cfg.translation_config.source_contrast, len(image_to_patches))
        else:
            image_dir = cfg.translation_config.output_dir
            os.makedirs(
                image_dir,
                exist_ok=True,
            )

        translated_paths_to_original_paths = {}
        translated_paths_to_original_paths_labels = {}
        for i, (original_path, patch_info_dict) in tqdm(enumerate(metadata['image_to_patches'].items()), desc="Reconstructing images from patches", total=len(metadata['image_to_patches'])):
            full_image = reconstruct_image_from_patches(
                patch_info_dict,
                metadata["patch_dimension"],
                dir_key="patch_dir",
                location_key="image_location"
            )
            match = re.match(r".*?_(\d{3})_\d{4}\.png$", original_path)
            if match:
                id = int(match.group(1))
            else:
                id = i
            new_path_image = f"{image_dir}/SYNTH_{target_contrast}_{id:03d}_0000.png"
            translated_paths_to_original_paths[new_path_image] = original_path
            cv2.imwrite(new_path_image, full_image)
            if cfg.translation_config.is_nnunet_dir:
                full_label = reconstruct_image_from_patches(
                    patch_info_dict,
                    metadata["patch_dimension"],
                    dir_key="patch_label_dir",
                    location_key="label_location"
                )
                original_label_path = patch_info_dict[0]["label_path_original"]
                new_label_path = f"{labels_dir}/SYNTH_{target_contrast}_{id:03d}.png"
                translated_paths_to_original_paths_labels[new_label_path] = original_label_path
                cv2.imwrite(new_label_path, full_label)

        # Save the translated paths to the original paths as a JSON file
        json_path = os.path.join(image_dir, "translated_to_original_paths.json")
        with open(json_path, 'w') as json_file:
            json.dump(translated_paths_to_original_paths, json_file, indent=4)
    finally:
        shutil.rmtree(image_dir_temp)
        shutil.rmtree(labels_dir_temp)


if __name__ == "__main__":
    translate_dataset()
