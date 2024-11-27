import functools
from collections import defaultdict

import e3nn
import numpy as np
import torch
import torchvision
from e3nn import o3
from torch import nn
from tqdm import tqdm

from cryoPARES.cacheManager import get_cache
from cryoPARES.configManager.config_searcher import inject_config
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import BATCH_PARTICLES_NAME
from cryoPARES.datamanager.datamanager import get_example_random_batch
from cryoPARES.geometry.metrics_angles import rotation_magnitude, mean_rot_matrix, rotation_error_rads
from cryoPARES.geometry.nearest_neigs_sphere import compute_nearest_neighbours
from cryoPARES.geometry.symmetry import getSymmetryGroup
from cryoPARES.models.image2sphere.imageEncoder.imageEncoder import ImageEncoder
from cryoPARES.models.image2sphere.so3Components import S2Conv, SO3Conv, I2SProjector, SO3OuptutGrid, SO3Activation


@inject_config()
class Image2Sphere(nn.Module):
    '''
    Instantiate Image2Sphere-style network for predicting distributions over SO(3) from
    single image
    '''

    cache = get_cache(cache_name=__qualname__)

    def __init__(self,
                 lmax, symmetry, label_smoothing:float, enforce_symmetry=True, encoder=None):
        super().__init__()

        if encoder is None:
            encoder = ImageEncoder()
        self.encoder = encoder
        self.lmax = lmax
        self.label_smoothing = label_smoothing

        batch = get_example_random_batch()
        x = batch[BATCH_PARTICLES_NAME]
        out = self.encoder(x)

        self.projector = I2SProjector(fmap_shape=out.shape[1:], lmax=lmax)
        out = self.projector(out)

        self.s2_conv = S2Conv(f_in=out.shape[1], lmax=lmax)
        out = self.s2_conv(out)

        self.so3_act = SO3Activation(lmax=lmax)

        out = self.so3_act(out)

        self.so3_conv = SO3Conv(f_in=out.shape[1], lmax=lmax)
        self.so3_grid = SO3OuptutGrid(lmax=lmax)


        self.symmetry = symmetry.upper()
        self.has_symmetry = (self.symmetry != "C1")

        #TODO: The following needs to be refactored, since it is problematic with multigpu. We need to make sure that
        #TODO: They are precomputed
        self.enforce_symmetry = enforce_symmetry
        self._get_symmetry_equivalent_idxs() #what are the idxs that are equivalent under a symmetry group
        self.rotation_contraction_idxs() #indices that need to be averaged to make sure everybody in the symmetry group are o
        self._get_neigs_matrix()
        print(f"Image2Sphere initialized (output_rotmats:{self.so3_grid.output_rotmats.shape[0]})")



    def predict_wignerDs(self, x):
        """

        :param x: image, tensor of shape (B, c, L, L)
        :return: flatten so3 irreps
        """

        x = self.encoder(x)
        x = self.projector(x)
        x = self.s2_conv(x)
        x = self.so3_act(x)
        x = self.so3_conv(x)
        return x

    def _from_wignerD_to_logits(self, x):
        rotMat_logits = torch.matmul(x, self.so3_grid.output_wigners).squeeze(1)
        if self.enforce_symmetry:
            rotMat_logits = self.aggregate_symmetry(rotMat_logits)
        return rotMat_logits


    def from_wignerD_to_topKMats(self, wD, k):
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
            reduced_sym_selected_idxs, _ = self.rotation_contraction_idxs()
            if self.enforce_symmetry and reduced_sym_selected_idxs is not None:
                _rotMat_logits = rotMat_logits[:, reduced_sym_selected_idxs]
                _, pred_rotmat_id = torch.topk(_rotMat_logits, k=k, dim=-1, largest=True)
                pred_rotmat_id = reduced_sym_selected_idxs[pred_rotmat_id]
            else:
                _, pred_rotmat_id = torch.topk(rotMat_logits, k=k, dim=-1, largest=True)

            pred_rotmat = self.so3_grid.output_rotmats[pred_rotmat_id]
        return rotMat_logits, pred_rotmat_id, pred_rotmat


    def forward(self, img, k=1):
        '''

        :img: float tensor of shape (B, c, L, L)
        :k: int number of top K elements to consider
        '''
        wD = self.predict_wignerDs(img)
        rotMat_logits, pred_rotmat_id, pred_rotmat = self.from_wignerD_to_topKMats(wD, k)

        probs = nn.functional.softmax(rotMat_logits, dim=-1)
        maxprob = probs.gather(dim=-1, index=pred_rotmat_id)
        return wD, rotMat_logits, pred_rotmat_id, pred_rotmat, maxprob


    @functools.lru_cache(3) #TODO: lru_cache uses self for the catching
    def _forward_with_neigs_batch_dim_selector(self, batch_size, device):
        return torch.arange(batch_size, device=device)

    def forward_with_neigs(self, img, k=1):

        wD = self.predict_wignerDs(img)
        rotMat_logits, pred_rotmat_id, pred_rotmat = self.from_wignerD_to_topKMats(wD, k)
        # If self.enforce_symmetry == True, then pred_rotmat_id contain indices corresponding to a reduced area of the projection sphere that covers all the posible views
        # Thus, if self.enforce_symmetry == True, the top-K won't have K-picks of equivalent points according to the symmetry
        with torch.no_grad():
            probs = nn.functional.softmax(rotMat_logits, dim=-1)
            neigs = self._get_neigs_matrix()[pred_rotmat_id,:] #These are the neigs of the top-K predicted rotation matrices
            _probs = probs[self._forward_with_neigs_batch_dim_selector(neigs.shape[0], device=neigs.device)[:,None,None], neigs]
            maxprob = _probs.sum(-1, keepdim=True)
            _rotmats = self.so3_grid.output_rotmats[neigs, ...]
            _probs = _probs / maxprob
            pred_rotmat, _ = mean_rot_matrix(_rotmats, weights=_probs, dim=2, compute_dispersion=False)
            # from scipy.spatial.transform import Rotation as R
            # print(_probs.round(decimals=3))
            # print(R.from_matrix(_rotmats[torch.arange(4),_probs.argmax(-1)]).as_euler("ZYZ", degrees=True).round(3))
            # print(R.from_matrix(pred_rotmat).as_euler("ZYZ", degrees=True).round(3))
        return wD, rotMat_logits, pred_rotmat_id, pred_rotmat, maxprob.squeeze(-1)

    def _get_neigs_matrix(self, k=10):
        """

        :param k: The number of nearest neighbours to compute
        :return: Tesor Pxk, where P is the number of points in SO(3) and k is the number of nearest neigbors
        """
        neigs = getattr(self, "_neigs", None)
        if neigs is None:
            neigs = compute_nearest_neighbours(self.so3_grid.output_eulerRad_yxy.cpu(), k=k,
                                               cache_dir=main_config.cachedir, n_jobs=1)["nearest_neigbours"]
            self.register_buffer("_neigs", neigs)
        return self._neigs


    @functools.lru_cache(3) #TODO: lru uses self for catching
    def _get_ies_for_aggregate_symmetry(self, batch_size, device):
        ies = torch.arange(batch_size, device=device).unsqueeze(-1).expand(-1, self.so3_grid.output_rotmats.shape[0]).unsqueeze(-1)
        return ies

    def aggregate_symmetry(self, signal):
        """

        :param signal: (BxK), K is the number of pose pixels
        :return:
        """
        jes = self._get_symmetry_equivalent_idxs()
        if jes is None: #Then symmetry is C1
            return signal
        ies = self._get_ies_for_aggregate_symmetry(signal.shape[0], signal.device)
        return signal[ies, jes].sum(-1)

    def rotation_contraction_idxs(self):
        """

        :return:
            - The indices of the geometry-equivalent selected poses (~ N_poses/symmetry_order,) Not exactly equal to N_poses/symmetry_order due to discretization problems
            - The _old_idx_to_new_idx mapping
        """

        if not hasattr(self, "_selected_rotmat_idxs"):
            if not self.has_symmetry:
                self._selected_rotmat_idxs = None
                self._old_idx_to_new_idx = None
            else:
                def _compute_selected_rotmat_idxs(n_rotmats, symmetry): #We need to store symmetry in the cache, because  self._get_symmetry_equivalent_idxs() uses symmetry
                    assert n_rotmats == self.so3_grid.output_rotmats.shape[0]
                    assert symmetry == self.symmetry
                    equiv_idxs = self._get_symmetry_equivalent_idxs().squeeze(0)
                    magnitudes = rotation_magnitude(self.so3_grid.output_rotmats)

                    seen = set()
                    selected_idxs=[]
                    old_idx_to_new_idx = -999999 * torch.ones(n_rotmats, dtype=torch.int64)
                    current_n_added = -1
                    for i in range(n_rotmats):
                        added = False
                        candidates = sorted(equiv_idxs[i].tolist(), key=lambda ei: (magnitudes[ei].round(decimals=5), ei))
                        # print([(magnitudes[c].item(), c) for c in candidates])
                        # assert torch.isclose(rotation_error_rads(self.so3_grid.output_rotmats[candidates][0], self.so3_grid.output_rotmats[candidates][1]), torch.FloatTensor([torch.pi]), atol=0.01).all()
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
                                old_idx_to_new_idx[ei] = current_n_added
                        # else:
                        #     raise RuntimeError(f"Error, node not added {i}")

                    selected_rotmat_idxs = torch.as_tensor(selected_idxs, device=self.so3_grid.output_rotmats.device)

                    # print(f"selected_rotmat_idxs {selected_rotmat_idxs.shape} representing symmetry reduction")
                    # assert len(selected_idxs)*len(self.symmetryGroupMatrix) == n_rotmats, "Error, wrong number of selected idxs"

                    # assert (old_idx_to_new_idx>=0).all()
                    # _test_rotmats = torch.FloatTensor([[[-0.8899,  0.1786, -0.4198],
                    #                                      [ 0.1791,  0.9831,  0.0386],
                    #                                      [ 0.4196, -0.0408, -0.9068]],
                    #                                     [[ 0.5233,  0.6696, -0.5270],
                    #                                      [-0.6490,  0.7140,  0.2628],
                    #                                      [ 0.5522,  0.2045,  0.8082]],
                    #                                     [[-0.4651, -0.8846, -0.0331],
                    #                                      [-0.0264,  0.0513, -0.9983],
                    #                                      [ 0.8848, -0.4635, -0.0472]],
                    #                                     [[-0.5505, -0.8333,  0.0518],
                    #                                      [ 0.8330, -0.5441,  0.1000],
                    #                                      [-0.0552,  0.0982,  0.9936]],
                    #                                    [[0.6974, 0.2501, 0.6717],
                    #                                     [0.0267, -0.9456, 0.3243],
                    #                                     [0.7162, -0.2082, -0.6661]]
                    #                                    ])
                    # print(rotation_error_rads((self.symmetryGroupMatrix[None, ...] @ _test_rotmats[:, None, ...]).view(-1, 3, 3), self.so3_grid.output_rotmats.unsqueeze(1)).view(4, -1).min(-1))
                    # print(rotation_error_rads((self.symmetryGroupMatrix[None, ...] @ _test_rotmats[:, None, ...]).view(-1, 3, 3), self.so3_grid.output_rotmats[selected_rotmat_idxs].unsqueeze(1)).view(4, -1).min(-1))

                    return selected_rotmat_idxs, old_idx_to_new_idx

                rotContractCache = get_cache("i2s_sym_selected_cache")
                compute_selected_rotmat_idxs = rotContractCache.cache(_compute_selected_rotmat_idxs)
                selected_rotmat_idxs, old_idx_to_new_idx = compute_selected_rotmat_idxs(self.so3_grid.output_rotmats.shape[0],  self.symmetry)
                # print(f"selected_rotmat_idxs {selected_rotmat_idxs.shape}")
                self.register_buffer("_selected_rotmat_idxs", selected_rotmat_idxs)
                self.register_buffer("_old_idx_to_new_idx", old_idx_to_new_idx)

        return self._selected_rotmat_idxs, self._old_idx_to_new_idx


    def _get_symmetry_equivalent_idxs(self):
        """
        Sets the buffers:
                  self.symmetryGroupMatrix: The stack of symmetry matrices, shape (G,3,3)
                 self._sym_equiv: The array of equivalent idxs under symmetry, shape (full_so3_n_poses, G). For each pose,
                 we annotate the pose idxs that are equivalent according to the group symmetry.

        :param batch_size:
        :return: self._sym_equiv
        """

        #TODO: this is executed by all the workers, which is a waste (and perhaps dangerous for racing conditions)
        #TODO: It should be only executed in the rank 0, or distributed
        if not hasattr(self, "_sym_equiv"):
            if not self.has_symmetry:
                self._sym_equiv = None
                self.register_buffer("symmetryGroupMatrix", torch.eye(3).unsqueeze(0))

            else:

                def _computeSymIndices(total_rotmats, symmetry):

                    symmetryGroupMatrix = torch.stack([torch.FloatTensor(x)
                                                       for x in getSymmetryGroup(symmetry).as_matrix()])

                    matched_idxs = torch.empty(total_rotmats, symmetryGroupMatrix.shape[0], dtype=torch.int64)
                    ori_device = self.so3_grid.output_rotmats.device
                    self.so3_grid.cuda()
                    output_rotmats = self.so3_grid.output_rotmats
                    _batch_size = 512
                    if torch.cuda.is_available():
                        symmetryGroupMatrix = symmetryGroupMatrix.cuda()
                        matched_idxs = matched_idxs.cuda()
                        gpu_memory_gb = torch.cuda.get_device_properties(symmetryGroupMatrix.device).total_memory / 1e9
                        if gpu_memory_gb > 23.:
                            _batch_size = _batch_size//4
                        else: #TODO: find better ranges
                            _batch_size = _batch_size//8

                    # Process in batches
                    for start_idx in tqdm(range(0, total_rotmats, _batch_size), desc="Computing indices for symmetry contraction", disable=False):
                        end_idx = start_idx + _batch_size
                        batch_rotmats = output_rotmats[start_idx:end_idx]

                        # Expand rotation matrices for the current batch
                        expanded_rotmats = torch.einsum("gij,pjk->gpik", symmetryGroupMatrix, batch_rotmats)

                        # Compute matched indices for each symmetry operation
                        for i in range(symmetryGroupMatrix.shape[0]):
                            _, batch_matched_idxs = self.so3_grid.nearest_rotmat(expanded_rotmats[i, ...])
                            matched_idxs[start_idx:end_idx, i] = batch_matched_idxs

                    # for i in range(output_rotmats.shape[0]): assert torch.isclose(rotation_error_rads(output_rotmats[matched_idxs[i, 0]], output_rotmats[matched_idxs[i, 1]]), torch.FloatTensor([torch.pi]).to(output_rotmats.device), atol=0.01) #Test code for sym=c2
                    self.so3_grid.to(ori_device)
                    return symmetryGroupMatrix.cpu(), matched_idxs.unsqueeze(0).cpu()

                symCache = get_cache("i2s_sym_equiv_cache")
                computeSymIndices = symCache.cache(_computeSymIndices)
                symmetryGroupMatrix, _sym_equiv= computeSymIndices(self.so3_grid.output_rotmats.shape[0], self.symmetry)
                self.register_buffer("symmetryGroupMatrix", symmetryGroupMatrix)
                self.register_buffer("_sym_equiv", _sym_equiv)

        return self._sym_equiv


    def compute_probabilities(self, img, hp_order=None):

        if hp_order is None:
            so3_grid = self.so3_grid
        else:
            so3_grid = SO3OuptutGrid(self.lmax, self.hp_order)

        x = self.predict_wignerDs(img)
        logits = torch.matmul(x, so3_grid.output_wigners).squeeze(1)
        probs = nn.Softmax(dim=1)(logits)

        return probs, so3_grid.output_rotmats


    def simCLR_like_loss(self, wD): #TODO: implement this
        return 0.

    def forward_and_loss(self, img, gt_rot, per_img_weight=None):
        '''Compute cross entropy loss using ground truth rotation, the correct label
        is the nearest rotation in the spatial grid to the ground truth rotation

        :img: float tensor of shape (B, c, L, L)
        :gt_rotation: valid rotation matrices, tensor of shape (B, 3, 3)
        :per_img_weight: float tensor of shape (B,) with per_image_weight for loss calculation
        '''

        wD, rotMat_logits, pred_rotmat_ids, pred_rotmats, maxprobs = self.forward(img)

        contrast_loss =  self.simCLR_like_loss(wD)

        if self.symmetry != "C1":
            n_groupElems = self.symmetryGroupMatrix.shape[0]
            #Perform symmetry expansion
            gtrotMats = self.symmetryGroupMatrix[None, ...] @ gt_rot[:, None, ...]
            rotMat_gtIds = self.so3_grid.nearest_rotmat_idxs(gtrotMats.view(-1, 3, 3))[-1].view(rotMat_logits.shape[0], -1)
            target_he = torch.zeros_like(rotMat_logits)
            rows = torch.arange(rotMat_logits.shape[0]).view(-1, 1).repeat(1, n_groupElems)
            target_he[rows, rotMat_gtIds] = 1 / n_groupElems
            loss = nn.functional.cross_entropy(rotMat_logits, target_he, reduction="none", label_smoothing=self.label_smoothing)

            with torch.no_grad():
                error_rads = rotation_error_rads(gtrotMats.view(-1,3,3),
                                                 torch.repeat_interleave(pred_rotmats, n_groupElems, dim=0))
                error_rads = error_rads.view(-1, n_groupElems)
                error_rads = error_rads.min(1).values

        else:
            # find nearest grid point to ground truth rotation matrix
            rot_idx = self.so3_grid.nearest_rotmat_idxs(gt_rot)[-1]
            loss = nn.functional.cross_entropy(rotMat_logits, rot_idx, reduction="none", label_smoothing=self.label_smoothing)
            with torch.no_grad():
                error_rads = rotation_error_rads(gt_rot, pred_rotmats)

        if per_img_weight is not None:
            loss = loss * per_img_weight.squeeze(-1)
        loss = loss.mean()
        loss = loss + contrast_loss

        return (wD, rotMat_logits, pred_rotmat_ids, pred_rotmats, maxprobs), loss, error_rads
    
@functools.lru_cache(maxsize=None)
def create_extraction_mask(lmax, device_type='cuda'):
    """
    Create a boolean mask to extract middle columns (m'=0) from flattened Wigner-D matrices.
    This mask is created once and can be reused for all extractions.

    Args:
        lmax: Maximum degree l
        device_type: String indicating device type ('cuda' or 'cpu')
    """
    mask = torch.zeros(sum((2 * l + 1) ** 2 for l in range(lmax + 1)), dtype=torch.bool)
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
    main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim = 128
    main_config.models.image2sphere.so3components.i2sprojector.rand_fraction_points_to_project = 1.
    main_config.models.image2sphere.so3components.i2sprojector.hp_order = 2
    main_config.models.image2sphere.so3components.s2conv.hp_order = 2
    main_config.models.image2sphere.so3components.so3ouptutgrid.hp_order = 3

def _test():
    _update_config_for_test()
    b = 4
    imgs = get_example_random_batch(4)[BATCH_PARTICLES_NAME]

    model = Image2Sphere(symmetry="c2") #lmax=6
    model.eval()
    with torch.inference_mode():
        from scipy.spatial.transform import Rotation
        gt_rot = torch.from_numpy(Rotation.random(b).as_matrix().astype(np.float32))
        # wD, rotMat_logits, pred_rotmat, maxprob = model.forward(imgs)
        # wD, rotMat_logits, pred_rotmat_idxs, pred_rotmat, maxprob = model.forward(imgs, k=8)
        wD, rotMat_logits, pred_rotmat_idxs, pred_rotmat, maxprob = model.forward_with_neigs(imgs, k=1)


        print("logits", rotMat_logits.shape)
        print("pred_rotmat", pred_rotmat.shape)
        model.forward_with_neigs(imgs)
        probs, output_rotmats = model.compute_probabilities(imgs)
        plot_so3_distribution(probs[0], output_rotmats, gt_rotation=gt_rot[0])


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
    imgs = get_example_random_batch()[BATCH_PARTICLES_NAME]

    encoder = nn.Conv2d(imgs.shape[1], main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim,
                        kernel_size=1, padding="same")
    model = Image2Sphere(lmax=6, symmetry="C1", enforce_symmetry=False, encoder=encoder)
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


if __name__ == "__main__":
    # _test()
    _test_rotation_invariance()
    print("Done!")
