import functools
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from e3nn import o3
from torch import nn

from cryoPARES.cacheManager import get_cache

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import BATCH_PARTICLES_NAME, SCRIPT_ENTRY_POINT
from cryoPARES.datamanager.datamanager import get_example_random_batch
from cryoPARES.geometry.metrics_angles import mean_rot_matrix, rotation_error_rads
from cryoPARES.geometry.nearest_neigs_sphere import compute_nearest_neighbours
from cryoPARES.models.image2sphere.imageEncoder.imageEncoder import ImageEncoder
from cryoPARES.models.image2sphere.so3Components import S2Conv, SO3Conv, I2SProjector, SO3OutputGrid, SO3Activation
from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM


class Image2Sphere(nn.Module):
    '''
    Instantiate Image2Sphere-style network for predicting distributions over SO(3) from
    single image
    '''

    cache = get_cache(cache_name=__qualname__)

    @inject_defaults_from_config(main_config.models.image2sphere, update_config_with_args=False)
    def __init__(self, symmetry:str, lmax: int = CONFIG_PARAM(),
                 hp_order: int = CONFIG_PARAM(config=main_config.models.image2sphere.so3components.so3outputgrid),
                 label_smoothing: float = CONFIG_PARAM(),
                 num_augmented_copies_per_batch: Optional[int] = CONFIG_PARAM(config=main_config.datamanager),
                 enforce_symmetry: bool = CONFIG_PARAM(),
                 encoder: Optional[nn.Module] = None,
                 use_simCLR: bool = False,
                 average_neigs_for_pred: bool = CONFIG_PARAM(),
                 example_batch: Optional[Dict[str, Any]] = None):
        super().__init__()

        self.encoder = encoder if encoder is not None else ImageEncoder()

        self.symmetry = symmetry
        self.lmax = lmax
        self.hp_order_output = hp_order
        self.label_smoothing = label_smoothing
        self.num_augmented_copies_per_batch = num_augmented_copies_per_batch #TODO: refactor SimCLR
        self.use_simCLR = use_simCLR
        self.n_neigs_to_compute = main_config.models.image2sphere.n_neigs_to_compute

        if example_batch is None:
            example_batch = get_example_random_batch(1)
        x = example_batch[BATCH_PARTICLES_NAME]
        out = self.encoder(x)

        self.projector = I2SProjector(fmap_shape=out.shape[1:], lmax=lmax)
        out = self.projector(out)

        self.s2_conv = S2Conv(f_in=out.shape[1], lmax=lmax)
        out = self.s2_conv(out)

        self.so3_act = SO3Activation(lmax=lmax)

        out = self.so3_act(out)

        self.so3_conv = SO3Conv(f_in=out.shape[1], lmax=lmax)

        self.so3_grid = SO3OutputGrid(lmax=self.lmax, hp_order=self.hp_order_output, symmetry=self.symmetry)

        self.symmetry = symmetry.upper()
        self.has_symmetry = (self.symmetry != "C1")

        self.enforce_symmetry = enforce_symmetry
        self.average_neigs_for_pred = average_neigs_for_pred
        self._initialize_caches()
        self._initialize_neigs() # Register nearest neighbors buffer

        if average_neigs_for_pred:
            self.forward = self.forward_with_neigs
        else:
            self.forward = self.forward_standard

    @torch.jit.ignore
    def _initialize_neigs(self, k: Optional[int] = None):
        """Initialize nearest neighbors matrix."""
        if k is None:
            k = self.n_neigs_to_compute
        euler_angles = self.so3_grid.output_eulerRad_yxy.cpu()
        neigs = compute_nearest_neighbours(
                euler_angles,
                k=k,
                cache_dir=main_config.cachedir,
                n_jobs=1 #TODO: Use more jobs. Read from config the num of threads
            )["nearest_neigbours"]
        self.register_buffer("_neigs", neigs)
        return self._neigs

    def _initialize_caches(self):
        # self.register_buffer("_cached_batch_size_range", torch.tensor(-1, dtype=torch.int64), persistent=False)
        # self.register_buffer("_cached_batch_indices", torch.empty(0, dtype=torch.int64), persistent=False)

        batch_size = max(main_config.train.batch_size, main_config.inference.batch_size)
        self._max_indices_buffer_size = 2 * batch_size
        self.register_buffer("_indices_buffer",
                           torch.arange(self._max_indices_buffer_size, dtype=torch.long),
                           persistent=False)

    def predict_wignerDs(self, x):
        """

        :param x: image, tensor of shape (B, c, L, L)
        :return: flatten so3 irreps
        """

        x = self.encoder(x)
        x = self.projector(x)
        x = self.s2_conv(x) #Conv2 does not work the same as before
        x = self.so3_act(x)
        x = self.so3_conv(x)
        return x

    def _from_wignerD_to_logits(self, x):
        rotMat_logits = torch.matmul(x, self.so3_grid.output_wigners).squeeze(1)
        if self.enforce_symmetry:
            rotMat_logits = self.so3_grid.aggregate_symmetry(rotMat_logits)
        return rotMat_logits


    def from_wignerD_to_topKMats(self, wD, k:int):
        """

        :param wD: The wignerD matrices
        :param k: The number of top-K matrices to report
        :return:
            rotMat_logits: (BxP) The logits obtained from the wignerD matrices by projecting them to the SO(3) grid
            pred_rotmat_id: (BxK) The top-K rotation matrix idxs. They refer to the original idxs, not the subset selected according to symmetry reduction
            pred_rotmat: (BxKx3x3) The top-K rotation matrices. They refer to the original matrices, not the subset selected according to symmetry reduction
        """
        rotMat_logits = self._from_wignerD_to_logits(wD) #This has symmetry summed values (symmetry contraction)
        with torch.no_grad():
            if self.enforce_symmetry and self.so3_grid.has_symmetry:
                reduced_sym_selected_idxs = self.so3_grid.selected_rotmat_idxs
                _rotMat_logits = rotMat_logits[:, reduced_sym_selected_idxs]
                _, pred_rotmat_id = torch.topk(_rotMat_logits, k=k, dim=-1, largest=True)
                pred_rotmat_id = reduced_sym_selected_idxs[pred_rotmat_id]
            else:
                _, pred_rotmat_id = torch.topk(rotMat_logits, k=k, dim=-1, largest=True)

            pred_rotmat = self.so3_grid.output_rotmats[pred_rotmat_id]
        return rotMat_logits, pred_rotmat_id, pred_rotmat

    @torch.jit.export
    def forward_standard(self, img:torch.Tensor, top_k:int):
        '''

        :img: float tensor of shape (B, c, L, L)
        :top_k: int number of top K elements to return
        '''
        wD = self.predict_wignerDs(img)
        rotMat_logits, pred_rotmat_id, pred_rotmat = self.from_wignerD_to_topKMats(wD, top_k)

        probs = nn.functional.softmax(rotMat_logits, dim=-1)
        maxprob = probs.gather(dim=-1, index=pred_rotmat_id)
        return wD, rotMat_logits, pred_rotmat_id, pred_rotmat, maxprob

    @torch.jit.export
    def forward_with_neigs(self, img:torch.Tensor, top_k:int): #TODO: FORWARD WITH NEIGS NEEDS TO BE EXPOSED

        wD = self.predict_wignerDs(img)
        rotMat_logits, pred_rotmat_id, pred_rotmat = self.from_wignerD_to_topKMats(wD, top_k)
        # If self.enforce_symmetry == True, then pred_rotmat_id contain indices corresponding to a reduced area of the projection sphere that covers all the possible views
        # Thus, if self.enforce_symmetry == True, the top-K won't have K-picks of equivalent points according to the symmetry
        with torch.no_grad():
            probs = nn.functional.softmax(rotMat_logits, dim=-1)
            neigs = self._get_neigs_matrix()[pred_rotmat_id,:] #These are the neigs of the top-K predicted rotation matrices
            batch_indices = self._get_batch_indices(neigs.shape[0], neigs.device)
            _probs = probs[batch_indices[:, None, None], neigs]
            maxprob = _probs.sum(-1, keepdim=True)
            _rotmats = self.so3_grid.output_rotmats[neigs, ...]
            _probs = _probs / maxprob
            pred_rotmat, _ = mean_rot_matrix(_rotmats, weights=_probs, dim=2, compute_dispersion=False)
            # from scipy.spatial.transform import Rotation as R
            # print(_probs.round(decimals=3))
            # print(R.from_matrix(_rotmats[torch.arange(4),_probs.argmax(-1)]).as_euler("ZYZ", degrees=True).round(3))
            # print(R.from_matrix(pred_rotmat).as_euler("ZYZ", degrees=True).round(3))
        return wD, rotMat_logits, pred_rotmat_id, pred_rotmat, maxprob.squeeze(-1)

    # def __get_batch_indices(self, batch_size: int, device: torch.device) -> torch.Tensor:
    #     """
    #     Used for indexing in forward_with_neigs
    #     :param batch_size:
    #     :param device:
    #     :return:
    #     """
    #     if self._cached_batch_size_range != batch_size or self._cached_batch_indices.device != device:
    #         # Update cache
    #         indices = torch.arange(batch_size, device=device)
    #         self._cached_batch_indices = indices
    #         self._cached_batch_size_range.copy_(torch.tensor(batch_size, device=device))
    #     return self._cached_batch_indices

    def _get_batch_indices(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Memory-efficient batch indices using pre-allocated buffer"""
        if batch_size <= self._max_indices_buffer_size:
            # Use pre-allocated buffer
            if self._indices_buffer.device != device:
                self._indices_buffer = self._indices_buffer.to(device, non_blocking=True)
            return self._indices_buffer[:batch_size]
        else:
            # For very large batches, create new tensor
            return torch.arange(batch_size, device=device)

    def _get_neigs_matrix(self):
        """

        :param k: The number of nearest neighbours to compute
        :return: Tesor Pxk, where P is the number of points in SO(3) and k is the number of nearest neigbors
        """
        # neigs = getattr(self, "_neigs", None)
        # if neigs is None:
        #     self._initialize_neigs()
        return self._neigs


    def compute_probabilities(self, img, hp_order=None):

        if hp_order is None:
            so3_grid = self.so3_grid
        else:
            so3_grid = SO3OutputGrid(self.lmax, hp_order, symmetry=self.symmetry)

        x = self.predict_wignerDs(img)
        logits = torch.matmul(x, so3_grid.output_wigners).squeeze(1)
        probs = nn.Softmax(dim=1)(logits)

        return probs, so3_grid.output_rotmats


    def simCLR_like_loss(self, wD): #TODO: implement this
        return NotImplementedError()

    def forward_and_loss(self, img, gt_rotmat, per_img_weight=None, top_k:int=1):
        '''Compute cross entropy loss using ground truth rotation, the correct label
        is the nearest rotation in the spatial grid to the ground truth rotation

        :img: float tensor of shape (B, c, L, L)
        :gt_rotmat: float tensor of valid rotation matrices, tensor of shape (B, 3, 3)
        :per_img_weight: float tensor of shape (B,) with per_image_weight for loss calculation
        :top_k: int number of top K elements to return
        '''

        wD, rotMat_logits, pred_rotmat_ids, pred_rotmats, maxprobs = self.forward(img, top_k=top_k)

        if self.use_simCLR:
            contrast_loss = self.simCLR_like_loss(wD)
        else:
            contrast_loss = 0

        if self.has_symmetry:
            n_groupElems = self.so3_grid.symmetryGroupMatrix.shape[0]
            rows = torch.arange(rotMat_logits.shape[0]).view(-1, 1).repeat(1, n_groupElems)
            with torch.no_grad(): #TODO: Try to use error_rads as part of the loss function
                gtrotMats = self.so3_grid.symmetryGroupMatrix[None, ...] @ gt_rotmat[:, None, ...]
                rotMat_gtIds = self.so3_grid.nearest_rotmat_idx(gtrotMats.view(-1, 3, 3))[-1].view(
                    rotMat_logits.shape[0], -1)
                target_ohe = torch.zeros_like(rotMat_logits)
                target_ohe[rows, rotMat_gtIds] = 1 / n_groupElems
                error_rads = rotation_error_rads(gtrotMats.view(-1,3,3),
                                                 torch.repeat_interleave(pred_rotmats, n_groupElems, dim=0)[:,0,...])
                error_rads = error_rads.view(-1, n_groupElems)
                error_rads = error_rads.min(1).values
            loss = nn.functional.cross_entropy(rotMat_logits, target_ohe, reduction="none", label_smoothing=self.label_smoothing)

        else:

            with torch.no_grad():
                # find nearest grid point to ground truth rotation matrix
                rot_idx = self.so3_grid.nearest_rotmat_idx(gt_rotmat)[-1]
                #We will consider top1 only
                error_rads = rotation_error_rads(gt_rotmat, pred_rotmats[:,0,...])
            loss = nn.functional.cross_entropy(rotMat_logits, rot_idx, reduction="none",
                                               label_smoothing=self.label_smoothing)

        if per_img_weight is not None:
            loss = loss * per_img_weight.squeeze(-1)
        loss = loss
        loss = loss + contrast_loss

        return (wD, rotMat_logits, pred_rotmat_ids, pred_rotmats, maxprobs), loss, error_rads


@functools.lru_cache(maxsize=2)
def create_extraction_mask(lmax, device):
    """
    Create a boolean mask to extract middle columns (m'=0) from flattened Wigner-D matrices.
    This mask is created once and can be reused for all extractions. Used to get the spherical harmonics

    Args:
        lmax: Maximum degree l
        device_type: String indicating device type ('cuda' or 'cpu')
    """
    mask = torch.zeros(sum((2 * l + 1) ** 2 for l in range(lmax + 1)), dtype=torch.bool, device=device)
    idx = 0
    for l in range(lmax + 1):
        dim = 2 * l + 1
        # Calculate indices for the middle column (m'=0)
        middle_col_indices = torch.arange(idx + l, idx + dim ** 2, dim)
        mask[middle_col_indices] = True
        idx += dim ** 2
    return mask


def extract_sh_coeffs_fast(flat_wigner_d, lmax):
    """
    Efficiently extract spherical harmonic coefficients from flattened Wigner-D matrices
    using cached mask.
    """
    device_type = 'cuda' if flat_wigner_d.is_cuda else 'cpu'
    extraction_mask = create_extraction_mask(lmax, device_type).to(flat_wigner_d.device)
    return flat_wigner_d[..., extraction_mask]


def plot_so3_distribution(probs: torch.Tensor,
                          rots: torch.Tensor,
                          gt_rotation=None,
                          fig=None,
                          ax=None,
                          display_threshold_probability=0.000005,
                          show_color_wheel: bool = True,
                          canonical_rotation=torch.eye(3),
                          ):
    '''
    Taken from https://github.com/google-research/google-research/blob/master/implicit_pdf/evaluation.py
    '''
    import matplotlib.pyplot as plt
    cmap = plt.cm.hsv

    def _show_single_marker(ax, rotation, marker, edgecolors=True, facecolors=False):
        alpha, beta, gamma = o3.matrix_to_angles(rotation)
        color = cmap(0.5 + gamma.repeat(2) / 2. / np.pi)[-1]
        ax.scatter(alpha, beta - np.pi / 2, s=2000, edgecolors=color, facecolors='none', marker=marker, linewidth=5)
        ax.scatter(alpha, beta - np.pi / 2, s=1500, edgecolors='k', facecolors='none', marker=marker, linewidth=2)
        ax.scatter(alpha, beta - np.pi / 2, s=2500, edgecolors='k', facecolors='none', marker=marker, linewidth=2)

    if ax is None:
        fig = plt.figure(figsize=(8, 4), dpi=200)
        fig.subplots_adjust(0.01, 0.08, 0.90, 0.95)
        ax = fig.add_subplot(111, projection='mollweide')

    rots = rots @ canonical_rotation
    scatterpoint_scaling = 3e3
    alpha, beta, gamma = o3.matrix_to_angles(rots)

    # offset alpha and beta so different gammas are visible
    R = 0.02
    alpha += R * np.cos(gamma)
    beta += R * np.sin(gamma)

    which_to_display = (probs > display_threshold_probability)

    # Display the distribution
    ax.scatter(alpha[which_to_display],
               beta[which_to_display] - np.pi / 2,
               s=scatterpoint_scaling * probs[which_to_display],
               c=cmap(0.5 + gamma[which_to_display] / 2. / np.pi))
    if gt_rotation is not None:
        if len(gt_rotation.shape) == 2:
            gt_rotation = gt_rotation.unsqueeze(0)
        gt_rotation = gt_rotation @ canonical_rotation
        _show_single_marker(ax, gt_rotation, 'o')
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
        ax.set_yticklabels([])
        ax.set_xticklabels([r'90$\degree$', None,
                            r'180$\degree$', None,
                            r'270$\degree$', None,
                            r'0$\degree$'], fontsize=14)
        ax.spines['polar'].set_visible(False)
        plt.text(0.5, 0.5, 'Tilt', fontsize=14,
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes)

    plt.show()


def _update_config_for_test():
    main_config.models.image2sphere.lmax = 6
    main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim = 512
    main_config.models.image2sphere.so3components.i2sprojector.rand_fraction_points_to_project = 1.
    main_config.models.image2sphere.so3components.i2sprojector.hp_order = 2
    main_config.models.image2sphere.so3components.s2conv.hp_order = 2
    main_config.models.image2sphere.so3components.s2conv.f_out = 16
    main_config.models.image2sphere.so3components.so3outputgrid.hp_order = 3

    main_config.datamanager.particlesdataset.image_size_px_for_nnet = 224
    main_config.models.image2sphere.so3components.i2sprojector.rand_fraction_points_to_project = 1
    main_config.models.image2sphere.label_smoothing = 0.1

def _test():
    _update_config_for_test()
    b = 4
    example_batch = get_example_random_batch(b, n_channels=3, seed=42)
    imgs = example_batch[BATCH_PARTICLES_NAME]

    import torchvision
    encoder = nn.Sequential(*list(torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).children())[:-2])

    model = Image2Sphere(encoder=encoder, symmetry="C2", enforce_symmetry=True, example_batch=example_batch)
    model.eval()
    out = model(imgs, top_k=1)
    print(out[0].shape)
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, '/tmp/scripted_model.pt')
    model = scripted_model
    with torch.inference_mode():
        from scipy.spatial.transform import Rotation
        gt_rot = torch.from_numpy(Rotation.random(b, random_state=42).as_matrix().astype(np.float32))
        wD, rotMat_logits, pred_rotmat_idxs, pred_rotmat, maxprob1 = model.forward(imgs, top_k=2)
        wD2, rotMat_logits2, pred_rotmat_idxs2, pred_rotmat2, maxprob2 = model.forward_with_neigs(imgs, top_k=2)
        wD3, rotMat_logits3, pred_rotmat_idxs3, pred_rotmat3, maxprob3 = model.forward_standard(imgs, top_k=2)
        print(maxprob1.allclose(maxprob2), maxprob1.allclose(maxprob3), maxprob2.allclose(maxprob3))
        # (wD, rotMat_logits, pred_rotmat_ids, pred_rotmat, maxprobs), loss, error_rads = model.forward_and_loss(imgs, gt_rot)


        print("logits", rotMat_logits.shape)
        print("pred_rotmat", pred_rotmat.shape)
        fog_wn = model.forward_with_neigs(imgs, top_k=1)
        # probs, output_rotmats = model.compute_probabilities(imgs) #Not working in jitted model
        probs, output_rotmats = rotMat_logits.softmax(-1), model.so3_grid.output_rotmats
        plot_so3_distribution(probs[0], output_rotmats, gt_rotation=gt_rot[0])
        print("Done!")

def _test_rotation_invariance(n_samples=10):
    """
    Test to verify that spherical harmonic coefficients are invariant to image rotations of 90 degrees.
    """
    _update_config_for_test()

    def compute_relative_errors(coeffs1, coeffs2):
        """
        Compute relative errors between two sets of coefficients
        """
        error = torch.abs(coeffs1 - coeffs2) / (torch.abs(coeffs1) + 1e-6)
        return {
            'mean_error': error.mean().item(),
            'max_error': error.max().item(),
            'std_error': error.std().item(),
            'median_error': error.median().item(),
            'p90_error': torch.quantile(error, 0.9).item()}

    # Initialize model
    imgs = get_example_random_batch(1)[BATCH_PARTICLES_NAME]

    encoder = nn.Conv2d(imgs.shape[1], main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim,
                        kernel_size=1, padding="same")
    model = Image2Sphere(symmetry="D2", lmax=6, enforce_symmetry=False, encoder=encoder)
    model.eval()

    # Store all results
    rotation_errors = defaultdict(list)
    baseline_errors = []

    for sample_idx in range(n_samples):
        # Get two different random batches for baseline comparison
        batch1 = get_example_random_batch(1)
        batch2 = get_example_random_batch(1)
        img1 = batch1[BATCH_PARTICLES_NAME][0:1]
        img2 = batch2[BATCH_PARTICLES_NAME][0:1]

        # Test 1: Rotation Invariance
        from torchvision import transforms
        rotations = [
            transforms.functional.rotate(img1, angle)
            for angle in [0, 90, 180, 270]
        ]
        rotations = torch.stack(rotations)

        with torch.no_grad():
            # Get coefficients for rotated versions
            wDs_rot = [model.predict_wignerDs(rot_img) for rot_img in rotations]
            sh_coeffs_rot = [extract_sh_coeffs_fast(wd, model.lmax).squeeze(1) for wd in wDs_rot]

            # Get coefficients for different image (baseline)
            wD2 = model.predict_wignerDs(img2)
            sh_coeffs2 = extract_sh_coeffs_fast(wD2, model.lmax).squeeze(1)

        # Compare rotated versions to original
        base_coeffs = sh_coeffs_rot[0]
        for i, rot_coeffs in enumerate(sh_coeffs_rot[1:], 1):
            errors = compute_relative_errors(base_coeffs, rot_coeffs)
            for key, value in errors.items():
                rotation_errors[f"{i * 90}_{key}"].append(value)  # Changed key format

        # Baseline: Compare different images
        baseline = compute_relative_errors(base_coeffs, sh_coeffs2)
        baseline_errors.append(baseline)

    # Compute statistics
    def compute_stats(errors_list):
        stats = {}
        for key in errors_list[0].keys():
            values = [e[key] for e in errors_list]
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return stats

    # Calculate statistics for both rotation and baseline errors
    rotation_stats = {}
    for angle in [90, 180, 270]:
        # Collect all errors for this angle
        angle_errors = [{k.split('_', 1)[1]: v
                         for k, v in rotation_errors.items()
                         if k.startswith(f"{angle}_")}]
        rotation_stats[f"{angle}deg"] = compute_stats(angle_errors)

    baseline_stats = compute_stats(baseline_errors)

    print("\nRotation Invariance Test Results (with baseline comparison):")
    print("--------------------------------------------------------")
    print("\nBaseline (Different Images):")
    for metric, values in baseline_stats.items():
        print(f"\n{metric}:")
        for stat, val in values.items():
            print(f"  {stat}: {val:.6f}")

    print("\nRotation Results:")
    for angle, stats in rotation_stats.items():
        print(f"\n{angle}:")
        for metric, values in stats.items():
            print(f"  {metric}: {values}")

    print("\nInterpretation:")
    # Compare mean errors
    for angle, stats in rotation_stats.items():
        ratio = stats['mean_error']['mean'] / baseline_stats['mean_error']['mean']
        print(f"\n{angle} rotation mean error is {ratio:.3f}x smaller than baseline")
        max_ratio = stats['max_error']['mean'] / baseline_stats['max_error']['mean']
        print(f"{angle} rotation max error is {max_ratio:.3f}x smaller than baseline")

    return rotation_stats, baseline_stats

def _test2():
    _update_config_for_test()
    b = 4
    example_batch = get_example_random_batch(b, n_channels=3, seed=42)
    imgs = example_batch[BATCH_PARTICLES_NAME]

    import torchvision
    encoder = nn.Sequential(*list(torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).children())[:-2])

    model = Image2Sphere(encoder=encoder, symmetry="C1", enforce_symmetry=True, example_batch=example_batch)
    model.eval()
    out = model(imgs, top_k=1)
    print(out[4])
    out2 = model(imgs[:2], top_k=1)
    out3 = model(imgs[2:], top_k=1)
    print(out2[4])
    print(out3[4])

if __name__ == "__main__":
    _test()
    # _test_rotation_invariance()
    _test2()
    print("Done!")
