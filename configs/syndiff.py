from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NetworkDistribution:
    """
    Configuration for the network distribution in a distributed computing environment.
    """
    num_proc_node: int  # Number of processing nodes
    node_rank: int  # The rank of the node within the distributed network
    gpus: List[int]  # The GPUs available on the node
    master_address: str  # IP address of the master node
    port_num: str  # Port number used for communication between nodes


@dataclass
class ModelConfig:
    """
    Configuration for the model.
    """
    image_size: int  # Size of the images
    num_channels: int  # Number of channels in the input images
    num_channels_dae: int  # Number of channels in the Denoising Autoencoder (DAE)
    latent_dim: int  # Dimensionality of the latent space
    num_timesteps: int  # Number of timesteps for the diffusion process
    z_emb_dim: int  # Dimensionality of the latent z embedding
    t_emb_dim: int  # Dimensionality of the time embedding
    use_geometric: bool  # Whether to use geometric progression for variance schedule
    beta_min: float  # Minimum value of beta for the noise schedule
    beta_max: float  # Maximum value of beta for the noise schedule

    ch_mult: List[int]  # Multipliers for each channel in the network layers
    num_res_blocks: int  # Number of residual blocks in each resolution
    attn_resolutions: List[int]  # Resolutions at which to apply attention
    resblock_type: str  # Type of residual block (e.g., 'biggan', 'ddpm')
    progressive: str  # Type of progressive growing (e.g., 'none', 'output_skip')
    progressive_input: str  # Input progression mode (e.g., 'input_skip')
    progressive_combine: str  # Method to combine features in progressive growing
    fir: bool  # Whether to use Finite Impulse Response in up/downsampling
    fir_kernel: List[int]  # Kernel sizes for FIR up/downsampling
    skip_rescale: bool  # Whether to rescale the activations in skip connections
    resamp_with_conv: bool  # Whether to perform resampling using convolution
    conditional: bool  # Whether the model is conditional
    n_mlp: int  # Number of layers in the MLP for z conditioning
    ngf: int  # Number of generator filters

    embedding_type: str  # Type of embedding (e.g., 'positional', 'fourier')
    fourier_scale: float  # Scaling factor for Fourier features
    not_use_tanh: bool  # Whether to avoid using tanh activation at the output


@dataclass
class OptimizationConfig:
    """
    Configuration for the optimizer.
    """
    num_epoch: int  # Number of epochs to train for
    batch_size: int  # Batch size
    lr_g: float  # Learning rate for the generator
    lr_d: float  # Learning rate for the discriminator
    beta1: float  # Beta1 for Adam
    beta2: float  # Beta2 for Adam
    no_lr_decay: bool  # Whether to decay the learning rate
    use_ema: bool  # Whether to use exponential moving averages
    ema_decay: float  # EMA decay rate
    r1_gamma: float  # R1 regularization rate
    lazy_reg: int  # Frequency of lazy regularization
    lambda_l1_loss: float  # L1 loss weight
    dropout: float  # Dropout rate


@dataclass
class SynDiffTrainingConfig:
    """
    Configuration for the training.
    """
    # General Configuration
    resume: bool  # Whether to resume training from a checkpoint
    save_content: bool  # Whether to save the generated content
    save_content_every: int  # Frequency of saving the generated content
    save_ckpt_every: int  # Frequency of saving the checkpoint
    training_dataset_path: Optional[str]  # Path to the dataset to train on
    paired: bool  # Whether the dataset is paired
    optimization_config: OptimizationConfig  # Configuration for the optimizer


@dataclass
class TranslationConfig:
    """
    Configuration for translation.
    """
    path_dataset_to_translate: str  # Path to the dataset to translate
    which_epoch: int  # Which epoch to use for translation
    source_contrast: str  # Source contrast for translation (target contrast is inferred from the contrasts in the config)
    nnunet_dataset_id: int  # NNUNet dataset ID for the new dataset
    nnunet_dir: str  # NNUNet directory for the new dataset


@dataclass
class SyndiffConfig:
    """
    Configuration for the Syndiff model.
    """
    seed: int  # Seed for reproducibility
    exp: str  # Experiment name
    syndiff_results_path: Optional[str]  # Path to save results
    contrast1: str  # First contrast
    contrast2: str  # Second contrast
    network_distribution: NetworkDistribution  # Network distribution
    model_config: ModelConfig  # Configuration for the model
    training_config: Optional[SynDiffTrainingConfig]  # Configuration for the training
    translation_config: Optional[TranslationConfig]  # Configuration for translation
