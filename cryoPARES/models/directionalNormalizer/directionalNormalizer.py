import os
import torch
import torch.nn as nn
import numpy as np
import healpy as hp
from typing import Dict, Optional, Tuple, List, Union
from collections import defaultdict
import pickle

from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.models.image2sphere.so3Components import SO3OutputGrid


class DirectionalPercentileNormalizer(nn.Module):
    """
    Neural network module for computing directional percentiles on S2 space.

    This module normalizes prediction scores based on their orientation in S2 space,
    addressing the issue where prediction quality can vary by viewing direction.
    It can be attached to an existing neural network that predicts SO(3) indices.

    The normalization is based on computing per-cone statistics (median and MAD)
    and converting raw scores to Z-scores, making scores comparable across different
    orientations regardless of inherent direction-specific biases.

    Important assumptions:
    1. SO(3) indices are organized as consecutive in-plane rotations for each cone
    2. The formula cone_index = so3_index // n_psi is valid for the grid structure
    3. The in-plane rotation dimension has consistent size (n_psi) across all cones
    """
    @inject_defaults_from_config(default_config=main_config.models.directionalNormalizer)
    def __init__(self, symmetry: str, hp_order: int = CONFIG_PARAM()):
        super().__init__()

        self.hp_order = hp_order
        self.symmetry = symmetry.upper()

        # Initialize SO3 grid
        self.so3_grid = SO3OutputGrid(lmax=1, hp_order=self.hp_order, symmetry=self.symmetry) #lmax is irrelevant in here
        self.n_so3_pixels = self.so3_grid.output_rotmats.shape[0]
        # Calculate grid dimensions
        self.n_cones = hp.nside2npix(hp.order2nside(hp_order))
        self.n_psi = self.n_so3_pixels // self.n_cones

        # Register buffers for normalization parameters
        self.register_buffer(
            "medians",
            torch.full((self.n_cones,), float('nan'), dtype=torch.float32)
        )
        self.register_buffer(
            "mads",
            torch.full((self.n_cones,), float('nan'), dtype=torch.float32)
        )
        self.register_buffer(
            "global_median",
            torch.tensor(float('nan'), dtype=torch.float32)
        )
        self.register_buffer(
            "global_mad",
            torch.tensor(1.0, dtype=torch.float32)
        )

        self.is_fitted = False



    def so3_to_cone_ids(self, so3_indices: torch.Tensor) -> torch.Tensor:
        """
        Convert SO(3) indices to cone indices using integer division.

        This mapping assumes the SO(3) grid structure from so3_healpix_grid_equiangular
        where the full orientation space is organized as:

        - n_cones cone directions (alpha, beta pairs)
        - For each cone, n_psi in-plane rotations (gamma angles)
        - The SO(3) index increases sequentially, with all in-plane rotations
          for a cone stored consecutively before moving to the next cone

        Args:
            so3_indices: Tensor of SO(3) indices

        Returns:
            Tensor of cone indices
        """
        return so3_indices // self.n_psi

    def rotmats_to_cone_id(self, rotmats: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrices to cone indices.

        Args:
            rotmats: Tensor of rotation matrices

        Returns:
            Tensor of cone indices
        """
        _, so3_indices = self.so3_grid.nearest_rotmat_idx(rotmats, reduce_sym=True)

        if len(so3_indices.shape) > 1:
            so3_indices = so3_indices.view(-1)
            n_top = so3_indices.size(-1)
            so3_indices = so3_indices.view(-1, n_top)

        return self.so3_to_cone_ids(so3_indices)

    def fit(
            self,
            pred_rotmats: torch.Tensor,
            scores: torch.Tensor,
            gt_rotmats: Optional[torch.Tensor] = None,
            good_particles_percentile: float = 95.0, #TODO: Add this to config
            min_particles_per_cone: int = 10, #TODO: Add this to config
    ) -> None:
        """
        Estimate normalization parameters for each cone from a reference dataset.

        This method analyzes scores grouped by orientation (cone) to compute
        robust statistics that will be used for normalization during inference.

        When ground truth is available, it uses particles with correct orientations.
        When ground truth is unavailable, it uses top-scoring particles, assuming
        they are more likely to be correct.

        Args:
            pred_rotmats: Predicted SO(3) rotmats for particles. Shape Bx3x3
            scores: Prediction scores for particles
            gt_rotmats: Ground truth SO(3) rotmats (if available for training)
            good_particles_percentile: Percentile of particles to use when no ground truth
                                      Higher values mean only considering top-scored particles
            min_particles_per_cone: Minimum number of particles required for reliable statistics
                                   Cones with fewer particles will use global statistics
        """
        if pred_rotmats.shape == 4:
            assert pred_rotmats.size(1) == 1
            pred_rotmats = pred_rotmats[:,0,...]
        # Get cone indices for predictions
        cone_indices = self.rotmats_to_cone_id(pred_rotmats)

        # Get ground truth cone indices if available
        if gt_rotmats is not None:
            gt_cone_indices = self.rotmats_to_cone_id(gt_rotmats).cpu().numpy()
        else:
            gt_cone_indices = None

        # Move to CPU for numpy operations
        cone_indices_cpu = cone_indices.cpu().numpy()
        scores_cpu = scores.cpu().numpy()

        # Initialize cone statistics
        cone_stats = defaultdict(lambda: {
            'good_scores': [],
            'bad_scores': [],
            'all_scores': []
        })

        # Collect scores by cone
        for i in range(len(cone_indices_cpu)):
            cone_idx = cone_indices_cpu[i]
            score = scores_cpu[i]
            cone_stats[cone_idx]['all_scores'].append(score)

            if gt_cone_indices is not None:
                gt_cone_idx = gt_cone_indices[i]
                if cone_idx == gt_cone_idx:
                    cone_stats[cone_idx]['good_scores'].append(score)
                else:
                    cone_stats[cone_idx]['bad_scores'].append(score)

        # Compute statistics for each cone
        all_medians = []
        all_mads = []

        for cone_idx in range(self.n_cones):
            if cone_idx not in cone_stats:
                # No data for this cone
                self.medians[cone_idx] = float('nan')
                self.mads[cone_idx] = float('nan')
                continue

            if gt_cone_indices is not None and len(cone_stats[cone_idx]['good_scores']) >= min_particles_per_cone:
                # Use ground truth good scores
                scores_to_use = np.array(cone_stats[cone_idx]['good_scores'])
            else:
                # Use top percentile of all scores
                all_scores = np.array(cone_stats[cone_idx]['all_scores'])
                if len(all_scores) >= min_particles_per_cone and len(all_scores) > 1:
                    threshold = np.percentile(all_scores, good_particles_percentile)
                    good_mask = all_scores >= threshold
                    scores_to_use = all_scores[good_mask]
                else:
                    scores_to_use = all_scores

            if len(scores_to_use) == 0:
                self.medians[cone_idx] = float('nan')
                self.mads[cone_idx] = float('nan')
                continue

            # Compute robust statistics
            median = np.median(scores_to_use)
            mad = np.median(np.abs(scores_to_use - median))

            self.medians[cone_idx] = torch.FloatTensor([median])
            self.mads[cone_idx] = torch.FloatTensor([max(mad, 1e-8)])  # Avoid division by zero

            all_medians.append(median)
            all_mads.append(max(mad, 1e-8))

        # Compute global statistics for cones with insufficient data
        if len(all_medians) > 0:
            self.global_median = torch.tensor(np.nanmedian(all_medians), dtype=torch.float32)
            self.global_mad = torch.tensor(max(np.nanmedian(all_mads), 1e-8), dtype=torch.float32)

            # Fill NaN values with global statistics
            nan_mask = torch.isnan(self.medians)
            if nan_mask.any():
                self.medians[nan_mask] = self.global_median

            nan_mask = torch.isnan(self.mads) | (self.mads == 0)
            if nan_mask.any():
                self.mads[nan_mask] = self.global_mad

        self.is_fitted = True

    def forward(self, pred_rotmats: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply directional normalization to scores.

        Args:
            pred_rotmats: SO(3) rotmats
            scores: Raw prediction scores

        Returns:
            Normalized scores (Z-scores)
        """
        # cone_indices = self.so3_to_cone_ids(so3_indices)
        cone_indices = self.rotmats_to_cone_id(pred_rotmats)
        # Get normalization parameters for each cone
        medians = self.medians[cone_indices]
        mads = self.mads[cone_indices]

        # Compute Z-scores
        z_scores = (scores - medians) / mads

        return z_scores

    def save(self, path: str) -> None:
        """
        Save normalization parameters to a file.

        Args:
            path: File path to save parameters
        """
        torch.save(self, path)
        # state_dict = {
        #     'hp_order': self.hp_order,
        #     'n_psi': self.n_psi,
        #     'symmetry': self.symmetry,
        #     'medians': self.medians.cpu().numpy(),
        #     'mads': self.mads.cpu().numpy(),
        #     'global_median': self.global_median.cpu().item(),
        #     'global_mad': self.global_mad.cpu().item(),
        #     'is_fitted': self.is_fitted
        # }
        #
        # with open(path, 'wb') as f:
        #     pickle.dump(state_dict, f)

    @classmethod
    def load(
            cls,
            path: str,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> 'DirectionalPercentileNormalizer':
        """
        Load normalization parameters from a file.

        Args:
            path: File path to load parameters from
            device: Device to load parameters to

        Returns:
            Loaded DirectionalPercentileNormalizer
        """
        return torch.load(path, weights_only=False)


def _test():
    """Test function for the DirectionalPercentileNormalizer."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hp_order = 2

    # Generate test data
    from scipy.spatial.transform import Rotation
    n_particles = 500
    so3_rotmats = torch.FloatTensor(Rotation.random(n_particles).as_matrix()).to(device)
    scores = torch.randn(n_particles).to(device)
    so3_gtrotmats = so3_rotmats.clone()

    # Create and fit normalizer
    normalizer = DirectionalPercentileNormalizer(symmetry='d2', hp_order=hp_order)
    normalizer.to(device)
    normalizer.fit(so3_rotmats, scores, so3_gtrotmats)

    # Test forward pass
    data = {'pred_rotmats': so3_rotmats, 'scores': scores}
    normalized_data = normalizer.forward(**data)

    print(f"Normalized scores shape: {normalized_data.shape}")

    # Test save/load
    normalizer.save('/tmp/normalizer_params.torch')
    loaded_normalizer = DirectionalPercentileNormalizer.load('/tmp/normalizer_params.torch')
    normalizer.to(device)
    normalized_data2 = loaded_normalizer.forward(**data)
    print(f"Original and loaded normalizers produce identical results: {torch.allclose(normalized_data, normalized_data2)}")


if __name__ == '__main__':
    _test()