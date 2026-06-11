import torch
import torch.nn as nn
import numpy as np


class Linear2DCenterDynamics(nn.Module):
    # this is my hardcoded 2D linear dynamics with center behavior.
    def __init__(self):
        super().__init__()
        self.mu = nn.Parameter(torch.Tensor([[0.0, -10], [10, 0.0]])).float()

    def forward(self, y):
        # y: [b, x_dim]
        return y @ self.mu.T  # shape (x_dim, b)


class LorenzDynamics(nn.Module):
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
        super().__init__()

        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        self.state_size = 3

    def forward(self, y):
        # y: [b, x_dim]
        x, y_, z = y[:, 0], y[:, 1], y[:, 2]
        dx = self.sigma * (y_ - x)
        dy = x * (self.rho - z) - y_
        dz = x * y_ - self.beta * z
        return torch.stack([dx, dy, dz], dim=1)  # shape (x_dim, b)


class RosslerDynamics(nn.Module):
    def __init__(self, a=0.2, b=0.2, c=5.7):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

        self.state_size = 3

    def forward(self, y):
        # y: [b, 3]
        x, y_, z = y[:, 0], y[:, 1], y[:, 2]
        dx = -y_ - z
        dy = x + self.a * y_
        dz = self.b + z * (x - self.c)
        return torch.stack([dx, dy, dz], dim=1)  # shape: [b, 3]


class ThomasDynamics(nn.Module):
    def __init__(
        self,
        b=0.2,
    ):
        super().__init__()
        self.b = b
        # self.a = a
        # self.c = c

        self.state_size = 3

    def forward(self, y):
        # y: [b, 3]
        x, y_, z = y[:, 0], y[:, 1], y[:, 2]
        dx = torch.sin(y_) - self.b * x
        dy = torch.sin(z) - self.b * y_
        dz = torch.sin(x) - self.b * z
        return torch.stack([dx, dy, dz], dim=1)  # shape: [b, 3]


class HalvorsenDynamics(nn.Module):
    def __init__(
        self,
        a=1.89,
    ):
        super().__init__()
        self.a = a

        self.state_size = 3

    def forward(self, y):
        # y: [b, 3]
        x, y_, z = y[:, 0], y[:, 1], y[:, 2]
        dx = -self.a * x - 4 * y_ - 4 * z - y**2
        dy = -self.a * y_ - 4 * z - 4 * x - z**2
        dz = -self.a * z - 4 * x - 4 * y_ - x**2
        return torch.stack([dx, dy, dz], dim=1)  # shape: [b, 3]


class VanDerPolsDynamics(nn.Module):
    def __init__(
        self,
        mu=1.89,
    ):
        super().__init__()
        self.mu = mu

        self.state_size = 2

    def forward(self, y):
        # y: [b, 3]
        x, y_ = y[:, 0], y[:, 1]
        dx = y_
        dy = self.mu * (1 - x**2) * y_ - x
        return torch.stack([dx, dy], dim=1)  # shape: [b, 3]
        # dz = -self.a * z - 4 * x - 4 * y_ - x **2
        # return torch.stack([dx, dy, dz], dim=1)  # shape: [b, 3]


# class CoupledVanDerPolsDynamics(nn.Module):
#     def __init__(self,
#                  mu=1.89,
#                  nu= 1,
#                  ):
#         super().__init__()
#         self.mu = mu

#         self.state_size = 4

#     def forward(self, y):
#         # y: [b, 3]
#         x, y_, z, w = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
#         dx = y_
#         dy = self.mu * (1 - x ** 2) * y_ - x

#         dz = w
#         dw = self.mu * (1 - x ** 2) * y_ - x
#         return torch.stack([dx, dy], dim=1)  # shape: [b, 3]
#         # dz = -self.a * z - 4 * x - 4 * y_ - x **2
#         # return torch.stack([dx, dy, dz], dim=1)  # shape: [b, 3]


class HindmarshRoseDynamics(nn.Module):
    def __init__(self, a=1, b=3, c=1, d=5, r=5e-3, s=4, I=10, xr=-8 / 5):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.I = I
        self.s = s
        self.r = r
        self.xr = xr

        self.state_size = 3

    def forward(self, y):
        # y: [b, 3]
        x, y_, z = y[:, 0], y[:, 1], y[:, 2]

        dx = y_ - self.a * x**3 + self.b * x**2 - z + self.I
        dy = self.c - self.d * x**2 - y_
        dz = self.r * (self.s * (x - self.xr) - z)
        return torch.stack([dx, dy, dz], dim=1)  # shape: [b, 3]


class LinearDynamics(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.mu = nn.Parameter(torch.Tensor([[0.0, -0.1], [0.1, 0.0]])).float()

    def forward(self, y):
        # y: [b, x_dim]
        return y @ self.mu.T * 100  # shape (x_dim, b)


# ============================================ Diffusion Term ============================================


class IsotropicDiffusion(nn.Module):
    def __init__(self, state_size, sigma=0.5):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(state_size) * sigma).float()

    def forward(self, y):
        # y: [b, x_dim]
        # return self.sigma  * y  # shape (b, x_dim)
        return self.sigma.expand(y.shape)


class BaseSDE(torch.nn.Module):
    def __init__(
        self,
        dynamics_fn,
        diffusion_fn,
        noise_type="diagonal",
        sde_type="stratonovich",
        speed=1.0,
    ):
        super().__init__()
        self.dynamics = dynamics_fn
        self.diffusion = diffusion_fn
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.speed = speed

    def f(self, t, y):  # Drift
        return self.dynamics(y) * self.speed  # shape (batch_size, state_size)

    def g(self, t, y):  # Diffusion
        return self.diffusion(y)


DYNAMICS_FNS = {
    "lorenz": LorenzDynamics,
    "rossler": RosslerDynamics,
    "linear": LinearDynamics,
    "linear2d": Linear2DCenterDynamics,
    "thomas": ThomasDynamics,
    "halvorsen": HalvorsenDynamics,
    "hindmarsh-rose": HindmarshRoseDynamics,
    "vanderpols": VanDerPolsDynamics,
}

DIFFUSION_FNS = {
    "isotropic": IsotropicDiffusion,
}

# LinearSDE = BaseSDE(Linear2DCenterDynamics(),
#                     IsotropicDiffusion(2, sigma=0.1))

# LorenzSDE = BaseSDE(LorenzDynamics(),
#                     IsotropicDiffusion(3, sigma=0.5))
