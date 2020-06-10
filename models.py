import torch.nn as nn
import torch

def poly_scale(input, exp):
    input = torch.sign(input) * torch.abs(input)**exp
    return input



class ParetoGenStatic(nn.Module):
    def __init__(self, insz, hsz, outsz, alpha=1, xi=1, act=nn.ReLU):
        super().__init__()
        self.nn = Generator(insz, hsz, outsz, act=act)

        self.alpha = alpha
        self.xi = xi

    def forward(self, noise):

        noise = noise ** self.alpha
        output = self.nn(noise)
        return poly_scale(output, self.xi/self.alpha)


class Generator(nn.Module):

    def __init__(self, insz, hsz, outsz, act=nn.ReLU):
        super().__init__()

        main = nn.Sequential(
            nn.Linear(insz, hsz),
            act(),
            nn.Linear(hsz, hsz),
            act(),
            nn.Linear(hsz, hsz),
            act(),
            nn.Linear(hsz, outsz),
        )
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output

