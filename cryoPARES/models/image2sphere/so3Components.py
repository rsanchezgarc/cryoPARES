import e3nn
import torch

from typing import Tuple, Optional
import numpy as np
from e3nn import o3
from e3nn.o3._so3grid import flat_wigner
from torch import nn

from cryoPARES.cacheManager import get_cache
from cryoPARES.configManager.config_searcher import inject_config
from cryoPARES.geometry.grids import s2_healpix_grid, so3_near_identity_grid_cartesianprod, so3_healpix_grid
from cryoPARES.geometry.metrics_angles import nearest_rotmat_idx


def s2_irreps(lmax):
    return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])

def so3_irreps(lmax):
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


@inject_config()
class I2SProjector(nn.Module):
    '''Define orthographic projection from image space to half of sphere, returning spherical harmonics representation
    '''

    cache = get_cache(cache_name=__qualname__)

    def __init__(self,
                 fmap_shape: Tuple[int, int, int],
                 sphere_fdim: int,
                 lmax: int,
                 hp_order: int,
                 coverage: float,
                 sigma: float,
                 max_beta: float,
                 taper_beta: float,
                 rand_fraction_points_to_project: Optional[float],
                 ):
        """

        :param fmap_shape: shape of incoming feature map (channels, height, width)
        :param sphere_fdim:  number of channels of featuremap projected to sphere
        :param lmax: maximum degree of harmonics
        :param hp_order: recursion level of healpy grid where points are projected
        :param coverage: fraction of feature map that is projected onto sphere
        :param sigma: stdev of gaussians used to sample points in image space
        :param max_beta: maximum azimuth angle projected onto sphere (np.pi/2 corresponds to half sphere)
        :param taper_beta: if less than max_beta, taper magnitude of projected features beyond this angle
        :param rand_fraction_points_to_project: number of grid points used to perform projection, acts like dropout regularizer
        """
        super().__init__()
        self.lmax = lmax

        # point-wise linear operation to convert to proper dimensionality if needed
        if fmap_shape[0] != sphere_fdim:
            self.conv1x1 = nn.Conv2d(fmap_shape[0], sphere_fdim, 1)
        else:
            self.conv1x1 = nn.Identity()

        kernel_grid, xyz, data, sph_harmonics = self._compute_parameters(fmap_shape, lmax, coverage,
                                                                         sigma, max_beta, taper_beta, hp_order)
        self.kernel_grid = kernel_grid
        self.xyz = xyz

        self.weight = nn.Parameter(data=data, requires_grad=True)

        self.n_pts = self.weight.shape[-1]
        self.ind = torch.arange(self.n_pts)
        if rand_fraction_points_to_project is None or rand_fraction_points_to_project > 0:
            self.n_subset = None
        else:
            self.n_subset = int(rand_fraction_points_to_project * self.n_pts) + 1

        self.register_buffer("Y", sph_harmonics)

    @staticmethod
    @cache.cache()
    def _compute_parameters(fmap_shape, lmax, coverage,
                            sigma, max_beta, taper_beta, hp_order):

        kernel_grid = s2_healpix_grid(max_beta=max_beta, hp_order=hp_order)
        xyz = o3.angles_to_xyz(*kernel_grid)

        max_radius = torch.linalg.norm(xyz[:, [0, 2]], dim=1).max()
        sample_x = coverage * xyz[:, 2] / max_radius  # range -1 to 1
        sample_y = coverage * xyz[:, 0] / max_radius

        gridx, gridy = torch.meshgrid(2 * [torch.linspace(-1, 1, fmap_shape[1])], indexing='ij')
        scale = 1 / np.sqrt(2 * np.pi * sigma ** 2)
        data = scale * torch.exp(-((gridx.unsqueeze(-1) - sample_x).pow(2) \
                                 + (gridy.unsqueeze(-1) - sample_y).pow(2)) / (2 * sigma ** 2))
        data = data / data.sum((0, 1), keepdims=True)

        # apply mask to taper magnitude near border if desired
        betas = kernel_grid[1]
        if taper_beta < max_beta:
            mask = ((betas - max_beta) / (taper_beta - max_beta)).clamp(max=1).view(1, 1, -1)
        else:
            mask = torch.ones_like(data)

        data = (mask * data).to(torch.float32)
        sph_harmonics = o3.spherical_harmonics_alpha_beta(range(lmax + 1), * kernel_grid, normalization='component')
        return kernel_grid, xyz, data, sph_harmonics

    def forward(self, x):
        '''
            :x: float tensor of shape (B, C, H, W)
            :return: feature vector of shape (B,C,S) where S is number of sph harmonics coefficents and C is the number of channels
        '''
        x = self.conv1x1(x)

        if self.n_subset is not None: #TODO: This if-statement prevents compilation by breaking the controlflow. Implement as a nnModule
            ind = torch.randperm(self.n_pts)[:self.n_subset]
        else:
            ind = self.ind

        x = torch.einsum('bchw,...hwp->bcp', x, self.weight[:,:, ind]) #self.weight[..., ind])
        x = torch.relu(x)
        x = torch.einsum('ps,bcp->bcs', self.Y[ind], x) / ind.shape[0] ** 0.5
        return x


@inject_config()
class S2Conv(nn.Module):
    '''Define S2 group convolution which outputs signal over SO(3) irreps'''

    cache = get_cache(cache_name=__qualname__)

    def __init__(self,
                 f_in: int,
                 f_out: int,
                 lmax: int,
                 hp_order: int,
                ):
        '''
        :param f_in: feature dimensionality of input signal
        :param f_out: feature dimensionality of output signal
        :param lmax: maximum degree of harmonics
        :param hp_order: The hp_order for the grid of the kernel

        '''
        super().__init__()
        spherical_harmonics, w, lin  = self.build_components(f_in, f_out, lmax, hp_order)
        self.w = nn.Parameter(w)
        self.register_buffer("Y", spherical_harmonics)
        self.lin = lin


    @staticmethod
    @cache.cache()
    def build_components(f_in: int, f_out: int, lmax: int, hp_order: int):
        kernel_grid = s2_healpix_grid(hp_order, max_beta=np.inf)
        spherical_harmonics = o3.spherical_harmonics_alpha_beta(
            range(lmax + 1),
            *kernel_grid,
            normalization="component"
        )
        s2_ir = s2_irreps(lmax)
        so3_ir = so3_irreps(lmax)
        w = nn.Parameter(torch.randn(f_in, f_out, kernel_grid[0].shape[0]))
        lin = o3.Linear(s2_ir, so3_ir, f_in=f_in, f_out=f_out, internal_weights=False)
        return spherical_harmonics, w, lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        :param x: tensor of shape (B, f_in, (2*lmax+1)**2)
        :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
        '''
        psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
        return self.lin(x, weight=psi)


@inject_config()
class SO3Conv(nn.Module):
    '''SO3 group convolution'''

    cache = get_cache(cache_name=__qualname__)

    def __init__(self,
                 f_in: int,
                 f_out: int,
                 lmax: int,
                 max_rads: float,
                 n_angles: int
                 ):
        '''
        :param f_in: feature dimensionality of input signal
        :param f_out: feature dimensionality of output signal
        :param lmax: maximum degree of harmonics for input/output signals
        :param max_rads: maximum angular distance from the identity in radians
        :param n_angles: number of euler angles (components) to sample within the max_rads. The final number of rotations will be n_angles**3

        '''
        super().__init__()
        # print("building SO3Conv")

        w, f_wigner, lin = SO3Conv.build_components(f_in, f_out, lmax, max_rads, n_angles)
        self.w = nn.Parameter(w)
        self.register_buffer("D", f_wigner)
        self.lin = lin
        # print("SO3Conv initialized")

    @staticmethod
    @cache.cache()
    def build_components(f_in: int, f_out: int, lmax: int, max_rads: float, n_angles: int):
        kernel_grid = so3_near_identity_grid_cartesianprod(max_rads, n_angles)
        f_wigner = flat_wigner(lmax, *kernel_grid)
        so3_ir = so3_irreps(lmax)
        w = nn.Parameter(torch.randn(f_in, f_out, kernel_grid[0].shape[0]))
        lin = o3.Linear(so3_ir, so3_ir, f_in=f_in, f_out=f_out, internal_weights=False)
        return w, f_wigner, lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        :param x: tensor of shape (B, f_in, sum_l^L (2*l+1)**2)
        :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
        '''
        psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x, weight=psi)



@inject_config()
class SO3OutputGrid(nn.Module):
    '''Define S2 group convolution which outputs signal over SO(3) irreps'''

    cache = get_cache(cache_name=__qualname__)

    def __init__(self,
                 lmax: int,
                 hp_order: int,
                ):
        '''
        :param lmax: maximum degree of harmonics
        :param hp_order: The hp_order for the grid of the kernel

        '''
        super().__init__()
        output_eulerRad_yxy, output_wigners, output_rotmats = self.build_components(lmax, hp_order)
        self.register_buffer("output_eulerRad_yxy", output_eulerRad_yxy)
        self.register_buffer("output_wigners", output_wigners)
        self.register_buffer("output_rotmats", output_rotmats)


    @staticmethod
    @cache.cache()
    def build_components(lmax: int, hp_order: int):
        output_eulerRad_yxy, _ = so3_healpix_grid(hp_order=hp_order)
        output_wigners = flat_wigner(lmax, *output_eulerRad_yxy).transpose(0, 1)
        output_rotmats = o3.angles_to_matrix(*output_eulerRad_yxy)
        return output_eulerRad_yxy, output_wigners, output_rotmats

    #TODO: Implement a nearest_rotmat that takes into account symmetry
    def nearest_rotmat(self, rotMat) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param rotMat: Tensor (Bx3x3)
        :return:
            - dot_trace: The similarity measurment fromthe rotMat and the highest score rotmat
            - idxs: The id of the closest rotMat
        """
        dot_trace, idxs = self.nearest_rotmat_idx(rotMat, self.output_rotmats)
        return dot_trace, self.output_rotmats[idxs]

    def nearest_rotmat_idxs(self, rotMat) -> Tuple[torch.Tensor, torch.Tensor]:
        return nearest_rotmat_idx(rotMat, self.output_rotmats)

    def forward(self, rotMat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.nearest_rotmat(rotMat)

@inject_config()
class SO3Activation(nn.Module):
    cache = get_cache(cache_name=__qualname__)
    def __init__(self, lmax, so3_act_resolution):
        super().__init__()
        self.lmax = lmax
        self.act = SO3Activation.build_components(lmax, so3_act_resolution)

    @staticmethod
    @cache.cache()
    def build_components(lmax: int, so3_act_resolution: int):
        return e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu, resolution=so3_act_resolution)

    def forward(self, x):
        return self.act(x)


def _test_SO3Conv():
    # Test parameters
    f_in, f_out, lmax = 16, 32, 4
    batch_size = 2

    # Initialize conv layer
    conv = SO3Conv(f_in=f_in, f_out=f_out, lmax=lmax)

    # Create sample input
    input_size = sum((2 * l + 1) ** 2 for l in range(lmax + 1))
    x = torch.randn(batch_size, f_in, input_size)

    # Test forward pass
    out1 = conv(x)
    out2 = torch.compile(conv)(x)
    assert torch.isclose(out1, out2, atol=1e-6).all()
    out3 = torch.jit.script(conv)(x)
    assert torch.isclose(out1, out2, atol=1e-6).all()
    assert torch.isclose(out2, out3, atol=1e-6).all()
    out4 = torch.jit.trace(conv, example_inputs=[x])(x)
    assert torch.isclose(out3, out4, atol=1e-6).all()

    # Test shapes
    output_size = sum((2 * l + 1) ** 2 for l in range(lmax + 1))
    assert out1.shape == (batch_size, f_out, output_size)

    print("SO3Conv tests passed")



def _test_S2Conv():
    # Test parameters
    f_in, f_out, lmax = 16, 32, 4
    batch_size = 2


    # Initialize conv layer
    conv = S2Conv(f_in=f_in, f_out=f_out, lmax=lmax, hp_order=3)

    # Create sample input
    input_size = sum((2 * l + 1) for l in range(lmax + 1))
    x = torch.randn(batch_size, f_in, input_size)

    # Test forward pass
    out1 = conv(x)
    out2 = torch.compile(conv)(x)
    assert torch.isclose(out1, out2, atol=1e-6).all()
    out3 = torch.jit.script(conv)(x)
    assert torch.isclose(out1, out2, atol=1e-6).all()
    assert torch.isclose(out2, out3, atol=1e-6).all()
    out4 = torch.jit.trace(conv, example_inputs=[x])(x)
    assert torch.isclose(out3, out4, atol=1e-6).all()

    # exported_program = torch.export.export(conv, args=(x,)) #This does not work because o3 uses torchscript

    # Test shapes
    output_size = sum((2 * l + 1) ** 2 for l in range(lmax + 1))
    assert out1.shape == (batch_size, f_out, output_size)


def _test_Image2SphereProjector():
    img_size = (1,32, 32)
    proj = I2SProjector(fmap_shape =(1,32, 32),
                         sphere_fdim = 16,
                         lmax = 4,
                         hp_order=3,
                        rand_fraction_points_to_project=1)
    img = torch.rand(*(1,)+img_size)
    out1 = proj(img)
    print(out1.shape)
    out2 = torch.compile(proj)(img)
    out3 = torch.jit.script(proj)(img)
    assert torch.isclose(out1, out2, atol=1e-7).all()
    assert torch.isclose(out2, out3, atol=1e-7).all()
    out4 = torch.jit.trace(proj, example_inputs=[img])(img)
    assert torch.isclose(out3, out4, atol=1e-7).all()



if __name__ == "__main__":
    # _test_Image2SphereProjector()
    # _test_S2Conv()
    _test_SO3Conv()