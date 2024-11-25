import torch
from e3nn.o3 import wigner_D, angles_to_matrix, matrix_to_angles
from e3nn.o3._so3grid import flat_wigner
import numpy as np


class WignerRotator(torch.nn.Module): #TODO: register rotation_wigners as a buffer
    def __init__(self, lmax):
        super().__init__()
        self.lmax = lmax
        
        # Track dimensions
        self.dims = [(2 * l + 1) for l in range(lmax + 1)]
        self.D = sum(self.dims)
        self.nFW = sum(d * d for d in self.dims)

        # Compute indices for flatten<->block conversion
        block_start = 0
        flat_start = 0
        rows = []
        cols = []

        for l, dim in enumerate(self.dims):
            for i in range(dim):
                for j in range(dim):
                    rows.append(block_start + i)
                    cols.append(block_start + j)
            block_start += dim
            flat_start += dim * dim

        self.register_buffer("flat_to_mat_row", torch.LongTensor(rows))
        self.register_buffer("flat_to_mat_col", torch.LongTensor(cols))

    def create_rotation_wigners(self, rotations):
        wigner_matrices = [wigner_D(l, *rotations) for l in range(self.lmax + 1)]
        return torch.stack([
            torch.block_diag(*[wigner_matrices[l][i] for l in range(self.lmax + 1)])
            for i in range(rotations.shape[1])
        ])

    def flatten_to_block(self, flat_coeffs):
        B = flat_coeffs.shape[0]
        block_coeffs = torch.zeros(B, self.D, self.D,
                                   device=flat_coeffs.device,
                                   dtype=flat_coeffs.dtype)
        block_coeffs[:, self.flat_to_mat_row, self.flat_to_mat_col] = flat_coeffs
        return block_coeffs

    def block_to_flatten(self, block_coeffs):
        return block_coeffs[..., self.flat_to_mat_row, self.flat_to_mat_col]

    def rotate(self, flat_coeffs, rotation_wigners):
        """
        Args:
            flat_coeffs: (B, nFW) tensor of flatten wigners
            rotation_wigners: (R, D, D) tensor of block diagonal wigners
        Returns:
            (B, R, nFW) tensor of rotated  flatten wigners
        """
        block_coeffs = self.flatten_to_block(flat_coeffs)  # (B, D, D)
        rotated_block = torch.einsum('rij,bjk->brik', rotation_wigners, block_coeffs)  # (B, R, D, D)
        return self.block_to_flatten(rotated_block)  # (B, R, nFW)




def compose_rotations(angles1, angles2):
    """Compose two rotations given as ZYZ Euler angles"""
    R1 = angles_to_matrix(*angles1)
    R2 = angles_to_matrix(*angles2)
    R_composed = R2 @ R1
    return matrix_to_angles(R_composed)



def run_rotation_tests():
    """Comprehensive test suite for WignerRotator"""

    test_cases = {
        'Basic Rotations': [
            {'lmax': 1, 'name': '90° Y rotation',
             'init': torch.zeros(3),
             'rot': torch.tensor([0.0, np.pi / 2, 0.0])},
            {'lmax': 2, 'name': '90° X rotation',
             'init': torch.zeros(3),
             'rot': torch.tensor([np.pi / 2, 0.0, 0.0])},
            {'lmax': 3, 'name': '90° Z rotation',
             'init': torch.zeros(3),
             'rot': torch.tensor([0.0, 0.0, np.pi / 2])},
        ],
        'Batch Tests': [
            {'lmax': 2, 'name': 'Multiple signals, single rotation',
             'batch_size': 3,
             'init': torch.tensor([np.pi / 6, np.pi / 4, np.pi / 3]),
             'rot': torch.tensor([0.0, np.pi / 2, 0.0])},
            {'lmax': 2, 'name': 'Single signal, multiple rotations',
             'init': torch.tensor([np.pi / 6, np.pi / 4, np.pi / 3]),
             'rots': torch.tensor([[0.0, np.pi / 2, 0.0],
                                   [np.pi / 2, 0.0, 0.0],
                                   [0.0, 0.0, np.pi / 2]]).T},
            {'lmax': 2, 'name': 'Multiple signals, multiple rotations',
             'batch_size': 2,
             'init': torch.tensor([np.pi / 6, np.pi / 4, np.pi / 3]),
             'rots': torch.tensor([[0.0, np.pi / 2, 0.0],
                                   [np.pi / 2, 0.0, 1.0]]).T},
        ]
    }

    for category, tests in test_cases.items():
        print(f"\n=== {category} ===")

        for test in tests:
            print(f"\nTesting: {test['name']}")

            rotator = WignerRotator(test['lmax'])

            # Handle batch dimension
            if 'batch_size' in test:
                # Create batch of signals
                signal = flat_wigner(test['lmax'], *test['init'])
                signal = signal.unsqueeze(0).expand(test['batch_size'], -1)
            else:
                signal = flat_wigner(test['lmax'], *test['init']).unsqueeze(0)

            if 'rots' in test:  # Multiple rotations
                rot_block_mat = rotator.create_rotation_wigners(test['rots'])
                rotated = rotator.rotate(signal, rot_block_mat)  # (B, R, nFW)

                # Check each batch and rotation combination
                for b in range(signal.shape[0]):
                    for r, rot in enumerate(test['rots'].T):
                        composed = compose_rotations(test['init'], rot)
                        expected = flat_wigner(test['lmax'], *composed)

                        success = torch.allclose(rotated[b, r], expected, atol=1e-3)
                        print(f"Batch {b}, Rotation {r}: {'✓' if success else '✗'}")
                        if not success:
                            print("Max difference:", torch.max(torch.abs(expected - rotated[b, r])))

            else:  # Single rotation
                rot_block_mat = rotator.create_rotation_wigners(test['rot'].unsqueeze(1))
                rotated = rotator.rotate(signal, rot_block_mat)  # (B, 1, nFW)

                composed = compose_rotations(test['init'], test['rot'])
                expected = flat_wigner(test['lmax'], *composed)

                # Check each batch
                for b in range(signal.shape[0]):
                    success = torch.allclose(rotated[b, 0], expected, atol=1e-3)
                    print(f"Batch {b}: {'✓' if success else '✗'}")
                    if not success:
                        print("Max difference:", torch.max(torch.abs(expected - rotated[b, 0])))


if __name__ == "__main__":
    run_rotation_tests()