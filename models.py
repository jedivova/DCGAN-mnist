# https://github.com/soumith/ganhacks
# https://github.com/AKASHKADEL/dcgan-mnist/blob/master/networks.py
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, outp_c=1, inp_c=100, ngf=32):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(inp_c, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf, outp_c, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.network(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, inp_c=1, ndf=32):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(

            nn.Conv2d(inp_c, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)