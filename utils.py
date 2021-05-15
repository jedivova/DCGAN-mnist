import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import math
import itertools
import imageio
import natsort
from glob import glob
import numpy as np
import random

def get_data_loader(batch_size):
    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, ), std=(0.5, ))])

    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def generate_images(epoch, path, fixed_noise, num_test_samples, netG, use_fixed=True):
    netG.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z = torch.randn(num_test_samples, fixed_noise.shape[1], 1, 1, device=device)
    size_figure_grid = int(math.sqrt(num_test_samples))
  
    if use_fixed:
        generated_fake_images = netG(fixed_noise)
        path += 'fixed_noise/'
        title = 'Fixed Noise'
    else:
        generated_fake_images = netG(z)
        path += 'variable_noise/'
        title = 'Variable Noise'
  
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6,6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    for k in range(num_test_samples):
        i = k//4
        j = k%4
        ax[i,j].cla()
        ax[i,j].imshow(generated_fake_images[k].data.cpu().numpy().reshape(28,28), cmap='Greys')
    label = 'Epoch_{}'.format(epoch+1)
    fig.text(0.5, 0.04, label, ha='center')
    fig.suptitle(title)
    fig.savefig(path+label+'.png')

def save_gif(path, fps, fixed_noise=False):
    if fixed_noise==True:
        path += 'fixed_noise/'
    else:
        path += 'variable_noise/'
    images = glob(path + '*.png')
    images = natsort.natsorted(images)
    gif = []

    for image in images:
        gif.append(imageio.imread(image))
    imageio.mimsave(path+'animated.gif', gif, fps=fps)



def smoothed_labels(batch_shape, lbl=1, max_smooth=0.2):
    if lbl==1:
        label = max_smooth * 2*(torch.rand(batch_shape)-0.5)+1
    else:
        label = max_smooth * torch.rand(batch_shape)
    return label


def set_all_seeds():
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_printoptions(precision=10)

    

    