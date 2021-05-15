import argparse
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models import Generator, Discriminator
from utils import get_data_loader, generate_images, save_gif, smoothed_labels, set_all_seeds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCGANS MNIST')
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--ndf', type=int, default=32, help='Number of features to be used in Discriminator network')
    parser.add_argument('--ngf', type=int, default=32, help='Number of features to be used in Generator network')
    parser.add_argument('--nz', type=int, default=100, help='Size of the noise')
    parser.add_argument('--d-lr', type=float, default=0.0002, help='Learning rate for the discriminator')
    parser.add_argument('--g-lr', type=float, default=0.0002, help='Learning rate for the generator')
    parser.add_argument('--nc', type=int, default=1, help='Number of input channels. Ex: for grayscale images: 1 and RGB images: 3 ')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-test-samples', type=int, default=16, help='Number of samples to visualize')
    parser.add_argument('--output-path', type=str, default='./results/', help='Path to save the images')
    parser.add_argument('--fps', type=int, default=25, help='frames-per-second value for the gif')
    parser.add_argument('--use-fixed', type=bool, default=True, help='Boolean to use fixed noise or not')

    opt = parser.parse_args()
    print(opt)

    set_all_seeds() #lock all seeds to get reproducible results
    writer = SummaryWriter() #tensorboard writer for logging
    writer_iter = 0
    # Gather MNIST Dataset    
    train_loader = get_data_loader(opt.batch_size)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    # Define Discriminator and Generator architectures
    netG = Generator(opt.nc, opt.nz, opt.ngf).to(device).train()
    netD = Discriminator(opt.nc, opt.ndf).to(device).train()

    # loss function
    criterion = nn.BCEWithLogitsLoss() # discriminator hasn't sigmoid layer

    # optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=opt.d_lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.g_lr, betas=(0.5, 0.999))
    
    # initialize other variables
    num_batches = len(train_loader)
    fixed_noise = torch.randn(opt.num_test_samples, opt.nz, 1, 1, device=device)

    for epoch in range(opt.num_epochs):
        netG.train(), netD.train()
        for i, (real_images, _) in enumerate(train_loader):
            bs = real_images.shape[0]
            ##############################
            #   Training discriminator   #
            ##############################

            netD.zero_grad()
            real_images = real_images.to(device)
            label = smoothed_labels(bs, lbl=1).to(device)

            output = netD(real_images)
            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.sigmoid().mean().item()


            noise = torch.randn(bs, opt.nz, 1, 1, device=device)
            fake_images = netG(noise)
            label = smoothed_labels(bs, lbl=0).to(device)

            output = netD(fake_images.detach())
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            D_G_z1 = output.sigmoid().mean().item()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            ##########################
            #   Training generator   #
            ##########################

            netG.zero_grad()
            label = smoothed_labels(bs, lbl=1).to(device)
            output = netD(fake_images)
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.sigmoid().mean().item()
            optimizerG.step()

            ##########################
            #         LOGGING        #
            ##########################

            losses_dict = {
                'D_loss_real': lossD_real.item(),
                'D_loss_fake': lossD_fake.item(),
                'D_loss': lossD.item(),
                'G_loss': lossG.item()
            }
            writer.add_scalars('Losses', losses_dict, writer_iter)

            accuracy_dict = {
                'D(x)': D_x,
                'Discriminator - D(G(x))': D_G_z1,
                'Generator - D(G(x))': D_G_z2
            }
            writer.add_scalars('mean_outp', accuracy_dict, writer_iter)
            writer_iter += 1

            if i % 50 == 0:
                print('Epoch [%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, opt.num_epochs, i, num_batches, lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))




        torch.save(netG.state_dict(), "results/checkpoints/generator_param.pth")
        torch.save(netD.state_dict(), "results/checkpoints/discriminator_param.pth")
        generate_images(epoch, opt.output_path, fixed_noise, opt.num_test_samples, netG, use_fixed=opt.use_fixed)

    # Save gif:
    save_gif(opt.output_path, opt.fps, fixed_noise=opt.use_fixed)