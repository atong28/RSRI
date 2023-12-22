import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import cv2
import os
from IPython.display import clear_output
from model import UNet, Hamiltonian

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################################
# Calculates the etas, alphas, and betas given a constant eta_const.                   #
########################################################################################
def get_parameters(eta_const, num_timesteps):
    etas = torch.ones(num_timesteps+1) * eta_const
    alphas = torch.cat((torch.tensor([1.0]), torch.cumprod(torch.cos(etas[:1000]), 0)))
    betas = 1 - (alphas ** 2)
    
    return etas, alphas, betas

########################################################################################
# Creates a dummy dataset of uniform values as images.                                 #
########################################################################################
class UniformDataset(Dataset):
    def __init__(self, num_samples, image_size=32, step_size=0.2):
        self.num_samples = num_samples
        self.image_size = image_size
        self.step_size = step_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random integer in the range [-5, 5] and multiply by step_size
        value = torch.randint(low=-5, high=6, size=(1,)).float() * self.step_size
        data_point = torch.ones((1, self.image_size, self.image_size)) * value

        # Classify into a class from 0 to 10 based on the generated value
        label = int((value + 1) / self.step_size)

        return data_point, label

########################################################################################
# Given a list of images (tensors) produce a grid with images. The image is saved to a #
# file if provided, and a title can be specified. For best results the length of the   #
# list should be a perfect square.                                                     #
########################################################################################
def show_images(images, file="", title=""):
    """Shows the provided images as sub-pictures in a square"""
    images = [im.permute(1,2,0).numpy() for im in images]

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx], cmap="gray", vmin=-1, vmax=1)
                plt.axis('off')
                idx += 1

    # Showing the figure
    fig.suptitle(title, fontsize=30)
    plt.show()
    if file != "":
        fig.savefig(file)

########################################################################################
# Generates images given the Hamiltonian model. Provide the number of images to        #
# generate, and also the number of color channels & size of image.                     #
########################################################################################   
def generate_image(hamiltonian, sample_size, channel, size):
    """Generate the image from the Gaussian noise"""

    frames = []
    hamiltonian.eval()
    with torch.no_grad():
        timesteps = list(range(hamiltonian.num_timesteps))[::-1]
        sample = torch.randn(sample_size, channel, size, size).to(device)
        
        for i, t in enumerate(tqdm(timesteps)):
            frame = []
            time_tensor = (torch.ones(sample_size,1) * t).long().to(device)
            residual = hamiltonian.reverse(sample, time_tensor)
            sample = hamiltonian.step(residual, time_tensor[0], sample)
            for i in range(sample_size):
                frame.append(sample[i].detach().cpu())
            frames.append(frame)
    return frames

########################################################################################
# Generates a video provided images, timesteps, and a file title and step size.        #
########################################################################################
def generate_video(images, num_timesteps, file_title='generation', step=5):
    # Generate Images with Matplotlib
    for i in range(0, num_timesteps, step):
        show_images(images[i], file=f'images/{i}.png', title=f'k={i}')
        clear_output(wait=True)

    # Create Frames List
    img_array = []
    for i in range(0, num_timesteps, 5):
        img = cv2.imread(f'images/{i}.png')
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    # Define VideoWriter
    out = cv2.VideoWriter(f'videos/{file_title}.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, size)

    # Generate Video
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    # Clear image directory to prevent clutter
    for root, dirs, files in os.walk('images'):
        for file in files:
            if file.endswith('.png'):
                os.remove(os.path.join(root, file))

########################################################################################
# Training loop for training the UNet/HGM.                                             #
########################################################################################
def training_loop(model, dataloader, optimizer, num_epochs, num_timesteps, device=device):
    """Training loop for DDPM"""

    global_step = 0
    losses = []

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            noise = torch.randn(batch.shape).to(device)
            timesteps = torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)

            noisy = model.add_noise(batch, noise, timesteps)
            noise_pred = model.reverse(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1

        progress_bar.close()
        clear_output(wait=True)

########################################################################################
# Save a a trained model to a .pt file.                                                #
########################################################################################
def save_model(model, filename):
    torch.save({
        'model_state_dict': model.state_dict()
    }, f'models/{filename}.pt')

########################################################################################
# Load a saved .pt model file.                                                         #
########################################################################################
def load_model(filename, num_timesteps, in_channels=1):
    network = UNet(in_channels=in_channels)
    network = network.to(device)
    model = Hamiltonian(network, num_timesteps, device=device)
    model.load_state_dict(torch.load(f'{filename}')['model_state_dict'])
    return model