import torch
from e3nn.o3 import wigner_D
import numpy as np


def create_rotation_block_matrix(lmax, rotations):
    """
    Create block diagonal rotation matrices for a set of rotations

    Args:
        lmax: Maximum degree L
        rotations: Tensor of shape (3, R) containing R sets of ZYZ Euler angles
    Returns:
        Block diagonal matrix of shape (R, D, D) where D is sum of (2l+1)
    """
    wigner_matrices = [wigner_D(l, *rotations) for l in range(lmax + 1)]
    block_matrices = torch.stack([
        torch.block_diag(*[wigner_matrices[l][i] for l in range(lmax + 1)])
        for i in range(rotations.shape[1])
    ])
    return block_matrices


def flatten_to_block(flat_coeffs, lmax):
    """
    Convert flat coefficients to block diagonal form

    Args:
        flat_coeffs: Tensor of shape (B, nFW)
        lmax: Maximum degree L
    Returns:
        Tensor of shape (B, D, D) where D is sum of (2l+1)
    """
    B = flat_coeffs.shape[0]
    block_coeffs = []
    start_idx = 0

    for l in range(lmax + 1):
        dim = 2 * l + 1
        size = dim * dim
        # Extract and reshape this l-block
        block = flat_coeffs[:, start_idx:start_idx + size].view(B, dim, dim)
        # Remove normalization for rotation
        block = block / (2 * l + 1) ** 0.5
        block_coeffs.append(block)
        start_idx += size

    return torch.stack([torch.block_diag(*[b[i] for b in block_coeffs])
                        for i in range(B)])


def block_to_flatten(block_coeffs, lmax):
    """
    Convert block diagonal form back to flat coefficients

    Args:
        block_coeffs: Tensor of shape (B, R, D, D)
        lmax: Maximum degree L
    Returns:
        Tensor of shape (B, R, nFW)
    """
    B, R = block_coeffs.shape[:2]
    flat_coeffs = []
    start_idx = 0

    for l in range(lmax + 1):
        dim = 2 * l + 1
        block = block_coeffs[..., start_idx:start_idx + dim, start_idx:start_idx + dim]
        # Restore normalization
        block = block * (2 * l + 1) ** 0.5
        flat_coeffs.append(block.reshape(B, R, -1))
        start_idx += dim

    return torch.cat(flat_coeffs, dim=-1)


def rotate_network_output(flat_coeffs, rotation_matrix, lmax):
    """
    Rotate the flat Wigner coefficients output by the network

    Args:
        flat_coeffs: Network output of shape (B, nFW)
        rotation_matrix: Block rotation matrices of shape (R, D, D)
        lmax: Maximum degree L
    Returns:
        Rotated coefficients of shape (B, R, nFW)
    """
    # Convert flat coefficients to block form
    block_coeffs = flatten_to_block(flat_coeffs, lmax)

    # Apply rotation
    rotated_block = torch.einsum('rij,bik->brkj', rotation_matrix, block_coeffs)

    # Convert back to flat form
    return block_to_flatten(rotated_block, lmax)


def test_rotation_network():
    # Setup
    lmax = 2
    batch_size = 3
    atol = 1e-3

    # Create test rotations
    angles_90y = torch.tensor([0.0, np.pi / 2, 0.0])
    angles_45y = torch.tensor([0.0, np.pi / 4, 0.0])
    test_rotations = torch.stack([angles_90y, angles_45y]).T  # Shape (3, R)

    # Create mock network output (normalized flat Wigner coefficients)
    mock_output = torch.stack([
        torch.cat([(2 * l + 1) ** 0.5 * wigner_D(l, *torch.zeros(3)).flatten(-2)
                   for l in range(lmax + 1)], dim=-1)
        for _ in range(batch_size)
    ])

    # Get rotation matrices
    rot_matrix = create_rotation_block_matrix(lmax, test_rotations)

    # Rotate network output
    rotated_output = rotate_network_output(mock_output, rot_matrix, lmax)

    # Test 1: Shape test
    nFW = sum((2 * l + 1) ** 2 for l in range(lmax + 1))
    assert rotated_output.shape == (batch_size, len(test_rotations.T), nFW)
    print("✓ Shape test passed")

    # Test 2: Compare with direct computation
    for b in range(batch_size):
        for r, angles in enumerate(test_rotations.T):
            expected = torch.cat([(2 * l + 1) ** 0.5 * wigner_D(l, *angles).flatten(-2)
                                  for l in range(lmax + 1)], dim=-1)
            assert torch.allclose(rotated_output[b, r], expected, atol=atol)
    print("✓ Rotation test passed")


if __name__ == "__main__":
    test_rotation_network()