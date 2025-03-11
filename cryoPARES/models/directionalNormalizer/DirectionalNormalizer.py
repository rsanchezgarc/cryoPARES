import os
import torch
import torch.nn as nn
import numpy as np
import healpy as hp
from typing import Dict, Optional, Tuple, List, Union
from collections import defaultdict
import pickle


class DirectionalPercentileNormalizer(nn.Module):
    """
    Neural network module for computing directional percentiles on SO(3) space.

    This module normalizes prediction scores based on their orientation in SO(3) space,
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

    def __init__(self,
                 hp_order: int,
                 n_psi: Optional[int] = None,
                 symmetry: str = "c1",
                 score_name: str = "score",
                 normalized_score_name: str = "normalized_score",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the DirectionalPercentileNormalizer.

        Args:
            hp_order: HEALPix order parameter defining the cone resolution
                      Higher values provide finer orientation binning
            n_psi: Number of in-plane rotations per cone
                  If None, calculated as 6 * (2**hp_order) based on standard grid
            symmetry: Symmetry group (e.g., "c1", "d2", etc.) for handling symmetric structures
            score_name: Name of the score field in the input dictionary
            normalized_score_name: Name of the normalized score field in the output dictionary
            device: Device to use for computation (cuda or cpu)
        """
        super().__init__()

        self.hp_order = hp_order
        self.symmetry = symmetry
        self.score_name = score_name
        self.normalized_score_name = normalized_score_name
        self.device = device

        # Calculate number of cones based on HEALPix order
        # This follows the HEALPix formula: npix = 12 * nside^2
        self.n_cones = hp.nside2npix(2 ** hp_order)

        # Calculate number of psi angles (in-plane rotations) if not provided
        # This assumes the standard grid spacing defined in so3_healpix_grid_equiangular
        if n_psi is None:
            self.n_psi = 6 * (2 ** hp_order)
        else:
            self.n_psi = n_psi

        # Total number of SO(3) pixels
        self.n_so3_pixels = self.n_cones * self.n_psi

        # Initialize parameters tensors to store statistics for each cone
        # Each cone will have two parameters:
        # - median: central tendency measure (robust to outliers)
        # - mad: median absolute deviation (robust measure of dispersion)
        self.register_buffer("medians", torch.full((self.n_cones,), float('nan'), dtype=torch.float32))
        self.register_buffer("mads", torch.full((self.n_cones,), float('nan'), dtype=torch.float32))

        # Global fallback parameters for cones with insufficient data
        self.register_buffer("global_median", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("global_mad", torch.tensor(1.0, dtype=torch.float32))

        # Track whether parameters have been estimated
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

    def fit(self,
            so3_indices: torch.Tensor,
            scores: torch.Tensor,
            gt_so3_indices: Optional[torch.Tensor] = None,
            good_particles_percentile: float = 0.9,
            min_particles_per_cone: int = 5) -> None:
        """
        Estimate normalization parameters for each cone from a reference dataset.

        This method analyzes scores grouped by orientation (cone) to compute
        robust statistics that will be used for normalization during inference.

        When ground truth is available, it uses particles with correct orientations.
        When ground truth is unavailable, it uses top-scoring particles, assuming
        they are more likely to be correct.

        Args:
            so3_indices: Predicted SO(3) indices for particles
            scores: Prediction scores for particles
            gt_so3_indices: Ground truth SO(3) indices (if available for training)
            good_particles_percentile: Percentile of particles to use when no ground truth
                                      Higher values mean only considering top-scored particles
            min_particles_per_cone: Minimum number of particles required for reliable statistics
                                   Cones with fewer particles will use global statistics
        """
        # Convert to cone indices
        cone_indices = self.so3_to_cone_ids(so3_indices)

        # If ground truth indices are provided, convert to cone indices too
        if gt_so3_indices is not None:
            gt_cone_indices = self.so3_to_cone_ids(gt_so3_indices)
        else:
            gt_cone_indices = None

        # Move to CPU for statistics calculation
        cone_indices_cpu = cone_indices.cpu().numpy()
        scores_cpu = scores.cpu().numpy()

        if gt_cone_indices is not None:
            gt_cone_indices_cpu = gt_cone_indices.cpu().numpy()

        # Collect statistics for each cone
        cone_stats = defaultdict(lambda: {"good_scores": [], "bad_scores": [], "all_scores": []})

        # Group scores by cone
        for i in range(len(cone_indices_cpu)):
            cone_idx = cone_indices_cpu[i]
            score = scores_cpu[i]

            cone_stats[cone_idx]["all_scores"].append(score)

            # If ground truth is available, separate good from bad predictions
            if gt_cone_indices is not None:
                gt_cone_idx = gt_cone_indices_cpu[i]
                if cone_idx == gt_cone_idx:
                    cone_stats[cone_idx]["good_scores"].append(score)
                else:
                    cone_stats[cone_idx]["bad_scores"].append(score)

        # Calculate statistics for each cone
        all_medians = []
        all_mads = []

        for cone_idx in range(self.n_cones):
            if cone_idx in cone_stats:
                # Determine which scores to use for statistics
                # Strategy 1: Use correctly predicted particles if ground truth available
                if gt_cone_indices is not None and len(cone_stats[cone_idx]["good_scores"]) >= min_particles_per_cone:
                    scores_to_use = np.array(cone_stats[cone_idx]["good_scores"])
                else:
                    # Strategy 2: Use top-scoring particles when no ground truth or insufficient good particles
                    all_scores = np.array(cone_stats[cone_idx]["all_scores"])
                    if len(all_scores) >= min_particles_per_cone:
                        if len(all_scores) > 1:
                            # Take top percentile of scores, assuming they're more likely correct
                            threshold = np.percentile(all_scores, 100 * good_particles_percentile)
                            scores_to_use = all_scores[all_scores >= threshold]
                        else:
                            scores_to_use = all_scores
                    else:
                        # Not enough particles for reliable statistics
                        self.medians[cone_idx] = float('nan')
                        self.mads[cone_idx] = float('nan')
                        continue

                # Calculate robust statistics: median and MAD
                median = np.median(scores_to_use)
                # MAD = median(|x - median(x)|)
                mad = np.median(np.abs(scores_to_use - median))

                # Store parameters
                self.medians[cone_idx] = median
                # Add small epsilon to avoid division by zero
                self.mads[cone_idx] = max(mad, 1e-6)

                # Collect for global statistics calculation
                all_medians.append(median)
                all_mads.append(mad)

        # Calculate global statistics for cones with insufficient data
        if all_medians:
            self.global_median = torch.tensor(np.nanmedian(all_medians), dtype=torch.float32)
            self.global_mad = torch.tensor(max(np.nanmedian(all_mads), 1e-6), dtype=torch.float32)

        # Fill in missing values with global statistics
        nan_mask = torch.isnan(self.medians)
        if nan_mask.any():
            self.medians[nan_mask] = self.global_median

        nan_mask = torch.isnan(self.mads) | (self.mads <= 0)
        if nan_mask.any():
            self.mads[nan_mask] = self.global_mad

        self.is_fitted = True

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Normalize scores based on directional statistics.

        Converts raw scores to Z-scores using the pre-computed median and MAD values
        for each orientation cone. This makes scores comparable across different
        orientations by accounting for direction-specific biases.

        Formula: z = (score - median) / MAD

        Args:
            data: Dictionary containing 'so3_indices' and score field
                 Must include the fields specified during initialization

        Returns:
            Dictionary with normalized scores added, preserving original entries
        """
        # Make a deep copy of the input dictionary
        output = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

        # Get SO(3) indices and scores
        so3_indices = output.get('so3_indices')
        scores = output.get(self.score_name)

        if so3_indices is None or scores is None:
            raise ValueError(f"Missing required fields: so3_indices or {self.score_name}")

        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before forward()")

        # Convert to cone indices
        cone_indices = self.so3_to_cone_ids(so3_indices)

        # Get statistics for each cone
        medians = self.medians[cone_indices]
        mads = self.mads[cone_indices]

        # Compute z-scores: (x - median) / MAD
        # This is a robust version of the standard z-score: (x - mean) / std
        z_scores = (scores - medians) / mads

        # Add normalized scores to output
        output[self.normalized_score_name] = z_scores

        return output

    def save(self, path: str) -> None:
        """
        Save normalization parameters to a file.

        Args:
            path: File path to save parameters
        """
        state_dict = {
            'hp_order': self.hp_order,
            'n_psi': self.n_psi,
            'symmetry': self.symmetry,
            'score_name': self.score_name,
            'normalized_score_name': self.normalized_score_name,
            'medians': self.medians.cpu().numpy(),
            'mads': self.mads.cpu().numpy(),
            'global_median': self.global_median.cpu().item(),
            'global_mad': self.global_mad.cpu().item(),
            'is_fitted': self.is_fitted
        }

        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)

    @classmethod
    def load(cls, path: str,
             device: str = "cuda" if torch.cuda.is_available() else "cpu") -> 'DirectionalPercentileNormalizer':
        """
        Load normalization parameters from a file.

        Args:
            path: File path to load parameters from
            device: Device to load parameters to

        Returns:
            Loaded DirectionalPercentileNormalizer
        """
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)

        # Create a new instance
        normalizer = cls(
            hp_order=state_dict['hp_order'],
            n_psi=state_dict['n_psi'],
            symmetry=state_dict['symmetry'],
            score_name=state_dict['score_name'],
            normalized_score_name=state_dict['normalized_score_name'],
            device=device
        )

        # Load parameters
        normalizer.medians = torch.tensor(state_dict['medians'], dtype=torch.float32, device=device)
        normalizer.mads = torch.tensor(state_dict['mads'], dtype=torch.float32, device=device)
        normalizer.global_median = torch.tensor(state_dict['global_median'], dtype=torch.float32, device=device)
        normalizer.global_mad = torch.tensor(state_dict['global_mad'], dtype=torch.float32, device=device)
        normalizer.is_fitted = state_dict['is_fitted']

        return normalizer


class SO3PredictorWithNormalization(nn.Module): #TODO: Check if this works
    """
    Combined model that integrates an SO(3) predictor with the DirectionalPercentileNormalizer.

    This wrapper makes it easy to use an existing SO(3) orientation predictor
    together with orientation-specific score normalization.
    """

    def __init__(self,
                 so3_predictor: nn.Module,
                 n_psi: Optional[int] = None,
                 score_name: str = "score",
                 normalized_score_name: str = "normalized_score"):
        """
        Initialize the combined model.

        Args:
            so3_predictor: Neural network that predicts SO(3) indices and scores
                          Must output a dictionary with 'so3_indices' and score_name fields
            n_psi: Number of in-plane rotations (if None, calculated based on hp_order)
            score_name: Name of the score field in the predictor output
            normalized_score_name: Name for the normalized score
        """
        super().__init__()

        self.so3_predictor = so3_predictor
        hp_order = self.so3_predictor.hp_order_output
        symmetry = self.so3_predictor.symmetry
        self.normalizer = DirectionalPercentileNormalizer(
            hp_order=hp_order,
            n_psi=n_psi,
            symmetry=symmetry,
            score_name=score_name,
            normalized_score_name=normalized_score_name
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the combined model.

        Args:
            x: Input tensor for the SO(3) predictor

        Returns:
            Dictionary with prediction results and normalized scores
        """
        # Get predictions from SO(3) predictor
        predictions = self.so3_predictor(x)

        # Apply normalization
        normalized_predictions = self.normalizer(predictions)

        return normalized_predictions

    def fit_normalizer(self,
                       so3_indices: torch.Tensor,
                       scores: torch.Tensor,
                       gt_so3_indices: Optional[torch.Tensor] = None) -> None:
        """
        Fit the normalizer parameters.

        This method should be called with a validation dataset before using
        the model for inference.

        Args:
            so3_indices: Predicted SO(3) indices
            scores: Prediction scores
            gt_so3_indices: Ground truth SO(3) indices (optional)
        """
        self.normalizer.fit(so3_indices, scores, gt_so3_indices)

    def save_normalizer(self, path: str) -> None:
        """
        Save the normalizer parameters to a file.

        Args:
            path: File path to save parameters
        """
        self.normalizer.save(path)

    def load_normalizer(self, path: str) -> None:
        """
        Load normalizer parameters from a file.

        Args:
            path: File path to load parameters from
        """
        loaded_normalizer = DirectionalPercentileNormalizer.load(path)
        self.normalizer = loaded_normalizer


# Example usage
if __name__ == "__main__":
    # Create some dummy data
    hp_order = 2
    n_cones = hp.nside2npix(2 ** hp_order)
    n_psi = 24  # Based on so3_healpix_grid_equiangular for hp_order=2
    n_so3_pixels = n_cones * n_psi

    # Pretend we have predictions for 1000 particles
    n_particles = 1000
    so3_indices = torch.randint(0, n_so3_pixels, (n_particles,))
    scores = torch.randn(n_particles)
    gt_so3_indices = torch.randint(0, n_so3_pixels, (n_particles,))

    # Create normalizer and fit
    normalizer = DirectionalPercentileNormalizer(hp_order=hp_order, n_psi=n_psi, symmetry="d2")
    normalizer.fit(so3_indices, scores, gt_so3_indices)

    # Apply normalization
    data = {
        'so3_indices': so3_indices,
        'score': scores
    }

    normalized_data = normalizer.forward(data)
    print(f"Normalized scores shape: {normalized_data['normalized_score'].shape}")

    # Save and load
    normalizer.save("normalizer_params.pkl")
    loaded_normalizer = DirectionalPercentileNormalizer.load("normalizer_params.pkl")

    # Test loaded normalizer
    normalized_data2 = loaded_normalizer.forward(data)
    print("Original and loaded normalizers produce identical results:",
          torch.allclose(normalized_data['normalized_score'], normalized_data2['normalized_score']))