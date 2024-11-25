import functools

import e3nn
import numpy as np
import torch
import torchvision
from e3nn import o3
from torch import nn
from tqdm import tqdm

from cryoPARES.cacheManager import get_cache
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.geometry.grids import s2_healpix_grid, so3_healpix_grid
from cryoPARES.geometry.metrics_angles import rotation_magnitude, mean_rot_matrix
from cryoPARES.geometry.nearest_neigs_sphere import compute_nearest_neighbours
from cryoPARES.geometry.symmetry import getSymmetryGroup
from cryoPARES.models.image2sphere.so3Components import S2Conv, SO3Conv, I2SProjector, SO3Grid


class I2S(nn.Module):
    '''
    Instantiate I2S-style network for predicting distributions over SO(3) from
    single image
    '''

    cache = get_cache(cache_name=__qualname__)

    def __init__(self,
                 image_encoder, imageEncoderOutputShape,
                 lmax, s2_fdim, so3_fdim,
                 hp_order_projector,
                 hp_order_s2,
                 hp_order_so3,
                 so3_act_resolution,  # TODO: what is the effect of resolution??
                 rand_fraction_points_to_project,
                 symmetry="c1",
                 enforce_symmetry=True):
        super().__init__()

        self.encoder = image_encoder
        self.lmax = lmax
        self.s2_fdim = s2_fdim
        self.so3_fdim = so3_fdim
        self.hp_order_projector = hp_order_projector
        self.hp_order_s2 = hp_order_s2
        self.hp_order = hp_order_so3

        self.projector = I2SProjector(
            fmap_shape=imageEncoderOutputShape,
            sphere_fdim=s2_fdim,
            lmax=lmax,
            hp_order=self.hp_order_projector,
            rand_fraction_points_to_project=rand_fraction_points_to_project
        )

        self.s2_conv = S2Conv(s2_fdim, so3_fdim, lmax, hp_order_s2)
        self.so3_act = I2S._build_so3_activation(lmax, so3_act_resolution)
        self.so3_conv = SO3Conv(so3_fdim, 1, lmax)
        self.so3_grid = SO3Grid(lmax, hp_order_so3)


        self.symmetry = symmetry
        self.has_symmetry = (symmetry.lower() != "c1")

        #TODO: The following needs to be refactored, since it is problematic with multigpu. We need to make sure that
        #TODO: They are precomputed
        self.enforce_symmetry = enforce_symmetry
        self._get_symmetry_equivalent_idxs() #what are the idxs that are equivalent under a symmetry group
        self.rotation_contraction_idxs() #indices that need to be averaged to make sure everybody in the symmetry group are o
        self._get_neigs_matrix()
        print(f"I2S initialized (output_rotmats:{self.so3_grid.output_rotmats.shape[0]})")



    @staticmethod
    @cache.cache()
    def _build_so3_activation(lmax, so3_act_resolution):
        return e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu, resolution=so3_act_resolution)

    def predict_wignerDs(self, x):
        '''Returns so3 irreps

        :x: image, tensor of shape (B, c, L, L)
        '''
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


    def forward(self, img, *, k=1):
        '''

        :img: float tensor of shape (B, c, L, L)
        :k: int number of top K elements to consider
        '''
        wD = self.predict_wignerDs(img)
        rotMat_logits, pred_rotmat_id, pred_rotmat = self.from_wignerD_to_topKMats(wD, k)

        probs = nn.functional.softmax(rotMat_logits, dim=-1)
        maxprob = probs.gather(dim=-1, index=pred_rotmat_id)
        return wD, rotMat_logits, pred_rotmat_id, pred_rotmat, maxprob


    @functools.lru_cache(3)
    def _forward_with_neigs_batch_dim_selector(self, batch_size, device):
        return torch.arange(batch_size, device=device)

    def forward_with_neigs(self, img, *, k=1):

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


    @functools.lru_cache(3)
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
            so3_grid = SO3Grid(self.lmax, self.hp_order)

        x = self.predict_wignerDs(img)
        logits = torch.matmul(x, so3_grid.output_wigners).squeeze(1)
        probs = nn.Softmax(dim=1)(logits)

        return probs, so3_grid.output_rotmats


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


def _test():
    model = torchvision.models.resnet152(weights=None)  # Better if pretrained
    model = nn.Sequential(*list(model.children())[:-2])

    b, c, l = 4, 3, 224
    imgs = torch.rand(b, c, l, l)

    model = I2S(model, model(imgs).shape[1:], lmax=6, s2_fdim=512, so3_fdim=16, hp_order_s2=1, hp_order_so3=3,
                hp_order_projector=2, so3_act_resolution=10, rand_fraction_points_to_project=0.5, symmetry="c2")
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


if __name__ == "__main__":
    _test()
    print("Done!")
