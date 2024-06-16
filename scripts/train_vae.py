"""
Script to train and preform inference on data.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from six.moves import xrange
import umap
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from sklearn.decomposition import PCA

from vqvae import VAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load Training Dataset
TRAIN_DIR = 'data_agg/train/'
training_data = ImageFolder(TRAIN_DIR, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0))
        ]))
training_loader = DataLoader(training_data, batch_size=16)

#Load Testing Dataset
TRAIN_DIR = 'data_agg/val/'
validation_data = ImageFolder(TRAIN_DIR, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0))
        ]))
validation_loader = DataLoader(validation_data)

def save_checkpoint(state, is_best, checkpoint_dir, best_model_dir):
    """
    Saves model in a directory with epoch, model and optimizer

    Parameters
    ----------
    is_best : boolean
        The boolean to indicate weather it's the best model
    checkpount_dir : str
        The directory to store checkpoint.
    best_model_dir : str
        The directory to store best model.
    """
    f_path = os.path.join(checkpoint_dir, "checkpoint_gen.pt")
    torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(best_model_dir, "best_model_gen.pt")
        shutil.copyfile(f_path, best_fpath)

def load_checkpoint(checkpoint_fpath, model, optimizer):
    """
    Loads model from the directory path provided.

    Parameters
    ----------
    checkpoint_fpath : str
        The directory where model is stored.

    Returns
    -------
    list of dict
        the stored model, optimizer and epoch
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['iteration']


# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--train", action='store_true' , help="To train the vae model")
parser.add_argument("-e", "--encode",action='store_true', help="To encode and decode data")
# parser.add_argument("-s", "--start", default=0, type=int, help="Starting offset to fit the ARIMA model")
parser.set_defaults(encode=True, train = False)
args = vars(parser.parse_args())
 
# Set up parameters
train = args["train"]
encode_data = args["encode"]
#Parameters 
batch_size = 256
num_training_updates = 15000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3
p=0 #to train or perform inference 

checkpoint_dir = "checkpoint/agg_chk" 
model_dir = "checkpoint/agg_chk" 

model = VAE(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
l1_loss = nn.L1Loss()
pca = PCA(n_components=3)
#Train Model
if train is True:
    print("Training Model")
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []

    for i in xrange(num_training_updates):
        (data, _) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_error = l1_loss(data_recon, data)
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()
        
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i+1) % 100 == 0:
            print('%d iterations' % (i+1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()

        is_best = False
        if i == 100:
            checkpoint_vae = {
                'iteration': i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(checkpoint_vae, is_best, checkpoint_dir, model_dir)
        
        if i == 7500:
            checkpoint_vae = {
                'iteration': i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(checkpoint_vae, is_best, checkpoint_dir, model_dir)
        if i == 15000:
            is_best = True
            checkpoint_vae = {
                'iteration': i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(checkpoint_vae, is_best, checkpoint_dir, model_dir)
        

   
    #Plot Loss    
    train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
    train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)
    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(1,2,1)
    ax.plot(train_res_recon_error_smooth)
    ax.set_yscale('Log')
    ax.set_title('L1 Loss')
    ax.set_xlabel('Iteration')

    ax = f.add_subplot(1,2,2)
    ax.plot(train_res_perplexity_smooth)
    ax.set_title('Perplexity')
    ax.set_xlabel('Iteration')

#Peform Reconstruction
if encode_data is True:
    print("Loading Checkpoint")
    checkpoint_path = "checkpoint/agg_chk/checkpoint_vae.pt"
    model, optimizer, num_training_updates= load_checkpoint(checkpoint_path, model, optimizer) 
    model.eval()
    c = 0
    for valid_originals, _ in validation_loader:
        valid_originals = valid_originals.to(device)

        vq_output_eval = model.encode(valid_originals)
        valid_quantize = model.quantize(vq_output_eval)
        valid_reconstructions = model.decode(valid_quantize)

        #Save Images
        latent_tensor = valid_quantize.view(valid_quantize.shape[1], valid_quantize.shape[2], valid_quantize.shape[3])
        latent = np.array(latent_tensor.cpu().detach().numpy())
        print(latent.shape) 
        latent_flat = latent.reshape(valid_quantize.shape[1], valid_quantize.shape[2] * valid_quantize.shape[3]).T
        print(latent_flat.shape)
        latent_pca = pca.fit_transform(latent_flat).T
        latent_pca1 =latent_pca.T 
        d1, d2 = latent_pca.shape
        pca_reshape = latent_pca.reshape(d1, valid_quantize.shape[2], valid_quantize.shape[3])
        pca_tensor = torch.Tensor(pca_reshape)
        print(pca_tensor.size())
        # reconstruction = valid_reconstructions.view(valid_reconstructions.shape[1], valid_reconstructions.shape[2], valid_reconstructions.shape[3])
        save_image(pca_tensor, 'results/Latent_%d'%c  + '.png', normalize=True)

        original = valid_originals.view(valid_originals.shape[1], valid_originals.shape[2], valid_originals.shape[3])
        save_image(original, 'results/Org_%d'%c  + '.png', normalize=True)

        recon_clip = torch.clamp(valid_reconstructions, min= 0.0, max = 1.0)
        reconstruction = valid_reconstructions.view(valid_reconstructions.shape[1], valid_reconstructions.shape[2], valid_reconstructions.shape[3])
        save_image(reconstruction, 'results/Recon_%d'%c  + '.png', normalize=True)
        c = c+1




