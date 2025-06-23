import functools

import e3nn
import torch

from typing import Tuple, Optional
import numpy as np
from e3nn import o3
from e3nn.o3._so3grid import flat_wigner
from torch import nn
from tqdm import tqdm

from cryoPARES.cacheManager import get_cache
from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.geometry.grids import s2_healpix_grid, so3_near_identity_grid_cartesianprod, so3_healpix_grid, \
    so3_near_identity_grid_ori
from cryoPARES.geometry.metrics_angles import nearest_rotmat_idx, rotation_magnitude
from cryoPARES.geometry.symmetry import getSymmetryGroup

GET_DEBUG_SEED = lambda: torch.Generator().manual_seed(42) # None

def s2_irreps(lmax):
    return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])

def so3_irreps(lmax):
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


class I2SProjector(nn.Module):
    '''Define orthographic projection from image space to half of sphere, returning spherical harmonics representation
    '''

    cache = get_cache(cache_name=__qualname__)
    @inject_defaults_from_config(main_config.models.image2sphere.so3components.i2sprojector, update_config_with_args=False)
    def __init__(self,
                 fmap_shape: Tuple[int, int, int],
                 sphere_fdim: int = CONFIG_PARAM(),
                 lmax: int = CONFIG_PARAM(config=main_config.models.image2sphere),
                 hp_order: int = CONFIG_PARAM(),
                 coverage: float = CONFIG_PARAM(),
                 sigma: float = CONFIG_PARAM(),
                 max_beta: float = CONFIG_PARAM(),
                 taper_beta: float = CONFIG_PARAM(),
                 rand_fraction_points_to_project: float = CONFIG_PARAM(),
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
        if rand_fraction_points_to_project is None or rand_fraction_points_to_project >= 1:
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

class S2Conv(nn.Module):
    '''Define S2 group convolution which outputs signal over SO(3) irreps'''

    cache = get_cache(cache_name=__qualname__)

    @inject_defaults_from_config(main_config.models.image2sphere.so3components.s2conv, update_config_with_args=False)
    def __init__(self,
                 f_in: int,
                 f_out: int = CONFIG_PARAM(),
                 lmax: int = CONFIG_PARAM(config=main_config.models.image2sphere),
                 hp_order: int = CONFIG_PARAM(),
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
        w = nn.Parameter(torch.randn(f_in, f_out, kernel_grid[0].shape[0], generator=GET_DEBUG_SEED()))
        lin = o3.Linear(s2_ir, so3_ir, f_in=f_in, f_out=f_out, internal_weights=False)
        return spherical_harmonics, w, lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        :param x: tensor of shape (B, f_in, (2*lmax+1)**2)
        :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
        '''
        psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class SO3Conv(nn.Module):
    '''SO3 group convolution'''

    cache = get_cache(cache_name=__qualname__)
    @inject_defaults_from_config(main_config.models.image2sphere.so3components.so3conv, update_config_with_args=False)
    def __init__(self,
                 f_in: int,
                 f_out: int = CONFIG_PARAM(),
                 lmax: int = CONFIG_PARAM(config=main_config.models.image2sphere),
                 max_rads: float = CONFIG_PARAM(),
                 n_angles: int = CONFIG_PARAM()
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
        # kernel_grid = so3_near_identity_grid_ori(n_alpha=8, n_beta=3) #TODO: This was the original implementation
        f_wigner = flat_wigner(lmax, *kernel_grid)
        so3_ir = so3_irreps(lmax)
        w = nn.Parameter(torch.randn(f_in, f_out, kernel_grid[0].shape[0], generator=GET_DEBUG_SEED()))
        lin = o3.Linear(so3_ir, so3_ir, f_in=f_in, f_out=f_out, internal_weights=False)
        return w, f_wigner, lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        :param x: tensor of shape (B, f_in, sum_l^L (2*l+1)**2)
        :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
        '''
        psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x, weight=psi)



class SO3OutputGrid(nn.Module):
    '''Define the output grid over SO(3) '''

    cache = get_cache(cache_name=__qualname__)
    @inject_defaults_from_config(main_config.models.image2sphere.so3components.so3outputgrid, update_config_with_args=False)
    def __init__(self,
                 lmax: int = CONFIG_PARAM(config=main_config.models.image2sphere),
                 hp_order: int = CONFIG_PARAM(),
                 symmetry: str = "C1",

                 ):
        '''
        :param lmax: maximum degree of harmonics
        :param hp_order: The hp_order for the grid of the kernel
        :param symmetry: The symmetry of the volume
        '''
        super().__init__()
        # print(f"Building SO3OutputGrid lmax: {lmax} ; hp_order: {hp_order}")
        self.lmax = lmax
        self.hp_order = hp_order

        self.symmetry = symmetry.upper()
        self.has_symmetry = self.symmetry != "C1"

        # output_eulerRad_yxy, output_wigners, output_rotmats = self.build_components(lmax, hp_order)
        (output_eulerRad_yxy, output_wigners, output_rotmats,
         symmetryGroupMatrix, sym_equiv_idxs,
         selected_rotmat_idxs, completeIdxs_to_reducedIdxs) = self.build_components(self.symmetry, lmax, hp_order)

        self.register_buffer("output_eulerRad_yxy", output_eulerRad_yxy)
        self.register_buffer("output_wigners", output_wigners)
        self.register_buffer("output_rotmats", output_rotmats)


        self.register_buffer("symmetryGroupMatrix", symmetryGroupMatrix) #Shape #1xoutput_rotmats.shape[0]xsymmetryGroupMatrix.shape[0]
        self.register_buffer("sym_equiv_idxs", sym_equiv_idxs)
        self.register_buffer("selected_rotmat_idxs", selected_rotmat_idxs) #Those are the indices of the rotmats that cover the portion of the projection sphere that corresponds to the symmetry
        self.register_buffer("completeIdxs_to_reducedIdxs", completeIdxs_to_reducedIdxs)

        self.register_buffer("_cached_batch_size_ies", torch.tensor(-1, dtype=torch.int64))
        self.register_buffer("_cached_ies", torch.empty(0, dtype=torch.int64))

    # @staticmethod
    # @cache.cache()
    # def build_components(lmax: int, hp_order: int):
    #     output_eulerRad_yxy, _ = so3_healpix_grid(hp_order=hp_order)
    #     output_wigners = flat_wigner(lmax, *output_eulerRad_yxy).transpose(0, 1)
    #     output_rotmats = o3.angles_to_matrix(*output_eulerRad_yxy)
    #     return output_eulerRad_yxy, output_wigners, output_rotmats

    @staticmethod
    @cache.cache()
    def build_components(symmetry: str, lmax: int, hp_order: int):
        output_eulerRad_yxy, _ = so3_healpix_grid(hp_order=hp_order)
        output_wigners = flat_wigner(lmax, *output_eulerRad_yxy).transpose(0, 1)
        output_rotmats = o3.angles_to_matrix(*output_eulerRad_yxy)

        (symmetryGroupMatrix, sym_equiv_idxs, selected_rotmat_idxs,
                        completeIdxs_to_reducedIdxs) = SO3OutputGrid.compute_symmetry_indices(output_rotmats, symmetry)

        return (output_eulerRad_yxy, output_wigners, output_rotmats, symmetryGroupMatrix,
                sym_equiv_idxs, selected_rotmat_idxs, completeIdxs_to_reducedIdxs)

    @staticmethod
    def compute_symmetry_indices(output_rotmats, symmetry) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute symmetry indices for the rotation matrices."""

        if symmetry == "C1":
            symmetryGroupMatrix = torch.eye(3).unsqueeze(0)
            sym_equiv_idxs = torch.arange(output_rotmats.shape[0])
            selected_rotmat_idxs = sym_equiv_idxs
            completeIdxs_to_reducedIdxs = sym_equiv_idxs
            return (symmetryGroupMatrix, sym_equiv_idxs.unsqueeze(0),
                    selected_rotmat_idxs, completeIdxs_to_reducedIdxs)

        n_rotmats = output_rotmats.shape[0]
        symmetryGroupMatrix = getSymmetryGroup(symmetry, as_matrix=True)

        sym_equiv_idxs = torch.empty(n_rotmats, symmetryGroupMatrix.shape[0], dtype=torch.int64)
        ori_device = output_rotmats.device

        batch_size = 512 if not torch.cuda.is_available() else (
            64 if torch.cuda.get_device_properties(0).total_memory / 1e9 > 23.0 else 32
        ) #TODO: Teak this numbers for better performance

        if torch.cuda.is_available():
            symmetryGroupMatrix = symmetryGroupMatrix.cuda()
            sym_equiv_idxs = sym_equiv_idxs.cuda()
            output_rotmats = output_rotmats.cuda()

        for start_idx in tqdm(range(0, n_rotmats, batch_size), desc=f"Computing symmetry indices {symmetry}"):
            end_idx = min(start_idx + batch_size, n_rotmats)
            batch_rotmats = output_rotmats[start_idx:end_idx]
            expanded_rotmats = torch.einsum("gij,pjk->gpik", symmetryGroupMatrix, batch_rotmats)

            for i in range(symmetryGroupMatrix.shape[0]):
                _, batch_matched_idxs = nearest_rotmat_idx(expanded_rotmats[i, ...], output_rotmats)
                sym_equiv_idxs[start_idx:end_idx, i] = batch_matched_idxs


        magnitudes = rotation_magnitude(output_rotmats)

        seen = set()
        selected_idxs = []
        completeIdxs_to_reducedIdxs = -9999999 * torch.ones(n_rotmats, dtype=torch.int64)
        current_n_added = -1

        for i in range(n_rotmats):
            added = False
            candidates = sorted(sym_equiv_idxs[i].tolist(),
                                key=lambda ei: (magnitudes[ei].round(decimals=5), ei))

            for ei in candidates:
                if ei in seen:
                    continue
                elif not added:
                    selected_idxs.append(ei)
                    added = True
                    current_n_added += 1
                seen.add(ei)

            if added:
                for ei in candidates:
                    completeIdxs_to_reducedIdxs[ei] = selected_idxs[-1]

        selected_rotmat_idxs = torch.as_tensor(selected_idxs, device=output_rotmats.device)

        return (symmetryGroupMatrix.cpu(), sym_equiv_idxs.unsqueeze(0).cpu(),
                selected_rotmat_idxs.cpu(), completeIdxs_to_reducedIdxs.cpu())

    def nearest_rotmat(self, rotMat) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param rotMat: Tensor (Bx3x3)
        :return:
            - dot_trace: The similarity measurement from the rotMat and the highest score rotmat
            - idxs: The id of the closest rotMat
        """
        dot_trace, idxs = self.nearest_rotmat_idx(rotMat)
        return dot_trace, self.output_rotmats[idxs]

    def nearest_rotmat_idx(self, rotMat, reduce_sym: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        dot_trace, idxs = nearest_rotmat_idx(rotMat, self.output_rotmats)
        if reduce_sym:
            idxs = self.completeIdxs_to_reducedIdxs[idxs]
        return dot_trace, idxs

    def symmetry_expand_rotmat_idx(self, idxs):
        expanded = self.sym_equiv_idxs[0, idxs]
        return expanded

    def symmetry_reduce_rotmat_idx(self, idxs):
        reduced = self.completeIdxs_to_reducedIdxs[idxs]
        return reduced

    def _get_ies_for_aggregate_symmetry(self, batch_size: int, device: torch.device) -> torch.Tensor:

        if self._cached_batch_size_ies != batch_size or self._cached_ies.device != device:
            # Update cache
            n_rotmats = self.output_rotmats.shape[0]
            ies = (torch.arange(batch_size, device=device)
                   .unsqueeze(-1)
                   .expand(-1, n_rotmats)
                   .unsqueeze(-1))
            self._cached_ies = ies
            self._cached_batch_size_ies.copy_(torch.tensor(batch_size, device=device))
        return self._cached_ies

    def aggregate_symmetry(self, signal):
        """

        :param signal: (BxK), K is the number of pose pixels
        :return:
        """
        if not self.has_symmetry:
            return signal
        jes = self.sym_equiv_idxs
        ies = self._get_ies_for_aggregate_symmetry(signal.shape[0], signal.device)
        return signal[ies, jes].sum(2)

    def forward(self, rotMat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.nearest_rotmat(rotMat)

class SO3Activation(nn.Module):
    cache = get_cache(cache_name=__qualname__)
    @inject_defaults_from_config(main_config.models.image2sphere.so3components.so3activation, update_config_with_args=False)
    def __init__(self,
                 lmax: int = CONFIG_PARAM(config=main_config.models.image2sphere),
                 so3_act_resolution: int = CONFIG_PARAM()):
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

def _test_so3grid():
    so3grid = SO3OutputGrid(symmetry="C2", lmax=6, hp_order=1)
    mat = torch.eye(3).unsqueeze(0)
    idx = so3grid.nearest_rotmat_idx(mat)
    print(idx)
    found = so3grid.nearest_rotmat(mat)
    print(found)
    avg_sig = so3grid.aggregate_symmetry(torch.rand(2, so3grid.output_rotmats.shape[0], 3))
    jitted = torch.jit.script(so3grid) #SO far is not compatible with torch.jit.script because of the dynamic method selection
    print(jitted)

if __name__ == "__main__":
    # _test_Image2SphereProjector()
    # _test_S2Conv()
    # _test_SO3Conv()
    _test_so3grid()