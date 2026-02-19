'''
Author: Julien Froustey
Copyright: GPLv3 (see LICENSE file)

Torch-native Box3D mixing heuristic utilities.
Designed to be called inside the model forward path and exported via TorchScript.
'''

import math
import torch
from torch import nn


class Box3DHeuristic(nn.Module):
    '''
    Implements a Box3D-like mixing baseline with a midpoint spherical quadrature.

    Inputs/outputs use normalized F4 convention:
      input  F4: [nsim, nu/nubar=2, flavor=NF, xyzt=4], with total density O(1)
      output F4: same shape
      output growth: [nsim] in normalized units (multiply by ntot for density units)
    '''

    def __init__(self, NF: int, resol_theta: int = 21, resol_phi: int = 41):
        super().__init__()
        if resol_theta < 3 or resol_phi < 3:
            raise ValueError("Box3D quadrature resolution must be at least 3 in each angular dimension")
        self.NF = int(NF)
        self.eps = 1.0e-12
        self.inv_4pi = 1.0 / (4.0 * math.pi)

        # Midpoint quadrature on (theta, phi)
        thetagrid = torch.linspace(0.0, math.pi, resol_theta, dtype=torch.float32)
        phigrid = torch.linspace(0.0, 2.0 * math.pi, resol_phi, dtype=torch.float32)
        thetamid = 0.5 * (thetagrid[:-1] + thetagrid[1:])
        phimid = 0.5 * (phigrid[:-1] + phigrid[1:])
        dtheta = thetagrid[1:] - thetagrid[:-1]
        dphi = phigrid[1:] - phigrid[:-1]

        sin_theta = torch.sin(thetamid)
        cos_theta = torch.cos(thetamid)
        sin_phi = torch.sin(phimid)
        cos_phi = torch.cos(phimid)

        vx = torch.outer(sin_theta, cos_phi).reshape(-1)
        vy = torch.outer(sin_theta, sin_phi).reshape(-1)
        vz = torch.outer(cos_theta, torch.ones_like(phimid)).reshape(-1)
        w = torch.outer(sin_theta * dtheta, dphi).reshape(-1)

        self.register_buffer("dir_x", vx)
        self.register_buffer("dir_y", vy)
        self.register_buffer("dir_z", vz)
        self.register_buffer("quad_w", w)

    def _z_approxbis(self, fluxfac: torch.Tensor) -> torch.Tensor:
        # Jedynak approximation:
        # p = 1 - (2(1-f)(1+1.01524f)) / (3 - 1.00651f^2 - 0.962251f^4 + 1.47353f^6 - 0.48953f^8)
        # Z = 2f / (1-p)
        f = torch.clamp(fluxfac, 0.0, 0.999999)
        f2 = f * f
        f4 = f2 * f2
        f6 = f4 * f2
        f8 = f4 * f4
        denom = 3.0 - 1.00651 * f2 - 0.962251 * f4 + 1.47353 * f6 - 0.48953 * f8
        p = 1.0 - (2.0 * (1.0 - f) * (1.0 + 1.01524 * f)) / torch.clamp(denom, min=self.eps)
        one_minus_p = torch.clamp(1.0 - p, min=1.0e-6)
        return (2.0 * f) / one_minus_p

    def _z_over_sinh_z(self, z: torch.Tensor) -> torch.Tensor:
        absz = torch.abs(z)
        # Stable near z=0: z/sinh(z) = 1 - z^2/6 + z^4/120 + O(z^6)
        series = 1.0 - (z * z) / 6.0 + (z * z * z * z) / 120.0
        direct = z / torch.sinh(z)
        return torch.where(absz < 1.0e-4, series, direct)

    def _extract_species_moments(self, F4: torch.Tensor):
        # Use {nue, nuebar, nux, nuxbar}, where heavy flavors are averaged.
        N_e = F4[:, 0, 0, 3]
        N_ebar = F4[:, 1, 0, 3]
        F_e = F4[:, 0, 0, :3]
        F_ebar = F4[:, 1, 0, :3]

        if self.NF > 1:
            N_x = torch.mean(F4[:, 0, 1:, 3], dim=1)
            N_xbar = torch.mean(F4[:, 1, 1:, 3], dim=1)
            F_x = torch.mean(F4[:, 0, 1:, :3], dim=1)
            F_xbar = torch.mean(F4[:, 1, 1:, :3], dim=1)
        else:
            N_x = N_e
            N_xbar = N_ebar
            F_x = F_e
            F_xbar = F_ebar

        N = torch.stack([N_e, N_ebar, N_x, N_xbar], dim=1)    # [B,4]
        F = torch.stack([F_e, F_ebar, F_x, F_xbar], dim=1)    # [B,4,3]
        return N, F

    def forward(self, F4_in: torch.Tensor):
        # F4_in: [B,2,NF,4] normalized
        B = F4_in.shape[0]
        dtype = F4_in.dtype
        device = F4_in.device

        N, F = self._extract_species_moments(F4_in)
        N = torch.clamp(N, min=self.eps)
        normF = torch.sqrt(torch.sum(F * F, dim=2))
        fluxfac = torch.clamp(normF / N, 0.0, 0.999999)
        Z = self._z_approxbis(fluxfac)

        dir_x = self.dir_x.to(device=device, dtype=dtype).view(1, 1, -1)
        dir_y = self.dir_y.to(device=device, dtype=dtype).view(1, 1, -1)
        dir_z = self.dir_z.to(device=device, dtype=dtype).view(1, 1, -1)
        w = self.quad_w.to(device=device, dtype=dtype).view(1, -1)

        mu_num = (
            F[:, :, 0].unsqueeze(-1) * dir_x
            + F[:, :, 1].unsqueeze(-1) * dir_y
            + F[:, :, 2].unsqueeze(-1) * dir_z
        )
        mu = mu_num / torch.clamp(normF.unsqueeze(-1), min=self.eps)

        z_over_sinh = self._z_over_sinh_z(Z.unsqueeze(-1))
        g = (N.unsqueeze(-1) * self.inv_4pi) * z_over_sinh * torch.exp(Z.unsqueeze(-1) * mu)  # [B,4,D]

        gnue = g[:, 0, :]
        gnuebar = g[:, 1, :]
        gnux = g[:, 2, :]
        gnuxbar = g[:, 3, :]

        delta = gnue - gnuebar - gnux + gnuxbar
        Hpos = (delta >= 0.0).to(dtype=dtype)
        Hneg = 1.0 - Hpos

        Iplus = torch.sum(w * torch.relu(delta), dim=1)
        Iminus = torch.sum(w * torch.relu(-delta), dim=1)
        Iplus_safe = torch.clamp(Iplus, min=self.eps)
        Iminus_safe = torch.clamp(Iminus, min=self.eps)

        coeff_pos = torch.where(
            Iplus < Iminus,
            torch.full_like(Iplus, 1.0 / 3.0),
            1.0 - (2.0 / 3.0) * (Iminus_safe / Iplus_safe),
        )
        coeff_neg = torch.where(
            Iplus < Iminus,
            1.0 - (2.0 / 3.0) * (Iplus_safe / Iminus_safe),
            torch.full_like(Iminus, 1.0 / 3.0),
        )
        coeff_pos = torch.clamp(coeff_pos, min=0.0, max=1.0)
        coeff_neg = torch.clamp(coeff_neg, min=0.0, max=1.0)

        Psur = coeff_pos.unsqueeze(1) * Hpos + coeff_neg.unsqueeze(1) * Hneg

        gnue_t = gnue * Psur + gnux * (1.0 - Psur)
        gnuebar_t = gnuebar * Psur + gnuxbar * (1.0 - Psur)
        gnux_t = 0.5 * (1.0 - Psur) * gnue + 0.5 * (1.0 + Psur) * gnux
        gnuxbar_t = 0.5 * (1.0 - Psur) * gnuebar + 0.5 * (1.0 + Psur) * gnuxbar
        g_t = torch.stack([gnue_t, gnuebar_t, gnux_t, gnuxbar_t], dim=1)

        newN = torch.sum(g_t * w.unsqueeze(1), dim=2)  # [B,4]
        newFx = torch.sum(g_t * w.unsqueeze(1) * dir_x, dim=2)
        newFy = torch.sum(g_t * w.unsqueeze(1) * dir_y, dim=2)
        newFz = torch.sum(g_t * w.unsqueeze(1) * dir_z, dim=2)
        newF = torch.stack([newFx, newFy, newFz], dim=2)  # [B,4,3]

        F4mix = torch.zeros_like(F4_in)
        # electron species
        F4mix[:, 0, 0, :3] = newF[:, 0, :]
        F4mix[:, 1, 0, :3] = newF[:, 1, :]
        F4mix[:, 0, 0, 3] = newN[:, 0]
        F4mix[:, 1, 0, 3] = newN[:, 1]

        # heavy flavors: identical x channels
        if self.NF > 1:
            F4mix[:, 0, 1:, :3] = newF[:, 2, :].unsqueeze(1).expand(B, self.NF - 1, 3)
            F4mix[:, 1, 1:, :3] = newF[:, 3, :].unsqueeze(1).expand(B, self.NF - 1, 3)
            F4mix[:, 0, 1:, 3] = newN[:, 2].unsqueeze(1).expand(B, self.NF - 1)
            F4mix[:, 1, 1:, 3] = newN[:, 3].unsqueeze(1).expand(B, self.NF - 1)

        growth = torch.sqrt(torch.clamp(Iplus * Iminus, min=0.0))
        return F4mix, growth
