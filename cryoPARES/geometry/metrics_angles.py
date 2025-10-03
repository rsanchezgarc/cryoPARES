from typing import Union, Tuple, Optional

import torch
from einops import rearrange, repeat

from cryoPARES.geometry.symmetry import getSymmetryGroup


def _compute_dot_trace(rotA, rotB):
    """

    :param rotA: tensors of shape (*,3,3)
    :param rotB: tensors of shape (*,3,3)
    :return: Tr(rotA, rotB.T)
    """
    prod = torch.matmul(rotA, rotB.transpose(-1, -2))
    trace = prod.diagonal(dim1=-1, dim2=-2).sum(-1)
    return trace

def rotation_magnitude(rot):
    """
    Compute rotation magnitude in radians.

    :param rot: tensor of shape (\*,3,3)
    :return: The rotation magnitude in radians
    """
    trace = rot.diagonal(dim1=-1, dim2=-2).sum(-1)
    return torch.arccos(torch.clamp((trace - 1) / 2, -1, 1))

def nearest_rotmat_idx(src, targets):
    """
    Return index of target that is nearest to each element in src.

    Uses negative trace of the dot product to avoid arccos operation.

    :param src: tensor of shape (B, 3, 3)
    :param targets: tensor of shape (\*, 3, 3)
    :return:
        - dot_trace: The value of the trace of the selected nearest rotmat idxs
        - idxs: The idxs of the nearest rotation matrices
    """
    trace = _compute_dot_trace(src.unsqueeze(1), targets.unsqueeze(0))
    dot_trace, idxs = torch.max(trace, dim=1)
    return dot_trace, idxs

#TODO: Implement the case that has symmetry for rotation_error_rads. You can call cryoPARES.geometry.symmetry.getSymmetryGroup(symmetry, as_matrix=True, device=device)
def rotation_error_rads(rotA, rotB):
    """
    Compute rotation error in radians between two rotation matrices.

    :param rotA: tensor of shape (\*,3,3). Rotation matrix
    :param rotB: tensor of shape (\*,3,3). Rotation matrix
    :return: rotation error in radians, tensor of shape (\*)
    """

    trace = _compute_dot_trace(rotA, rotB)
    return torch.arccos(torch.clamp((trace - 1) / 2, -1, 1))


def rotation_error_with_sym(rotA, rotB, symmetry=None):
    """
    Compute rotation error in radians between two rotation matrices with symmetry.

    :param rotA: tensor of shape (\*,3,3). Rotation matrix
    :param rotB: tensor of shape (\*,3,3). Rotation matrix
    :param symmetry: string or None. Symmetry group (e.g., 'I', 'O', 'T', 'D7', etc.)
                    If None, computes standard rotation error without symmetry
    :return: rotation error in radians, tensor of shape (\*)
    """

    if symmetry is None:
        # Standard case without symmetry
        return rotation_error_rads(rotA, rotB)

    else:
        # Case with symmetry - find minimum error over all symmetry operations
        device = rotA.device if hasattr(rotA, 'device') else None

        # Get symmetry group matrices
        sym_matrices = getSymmetryGroup(symmetry, as_matrix=True, device=device)  # Shape: (n_sym, 3, 3)

        batch_shape = rotA.shape[:-2]
        n_sym = sym_matrices.shape[0]

        # Reshape for efficient batch matrix multiplication
        # rotA: (*batch, 3, 3) -> (batch_flat, 3, 3)
        # rotB: (*batch, 3, 3) -> (batch_flat, 3, 3)
        rotA_flat = rearrange(rotA, '... i j -> (...) i j')
        rotB_flat = rearrange(rotB, '... i j -> (...) i j')
        batch_size = rotA_flat.shape[0]

        # Expand rotB to apply all symmetry operations at once
        # rotB_flat: (batch, 3, 3) -> (batch, n_sym, 3, 3)
        rotB_expanded = repeat(rotB_flat, 'b i j -> b s i j', s=n_sym)

        # Expand symmetry matrices: (n_sym, 3, 3) -> (batch, n_sym, 3, 3)
        sym_expanded = repeat(sym_matrices, 's i j -> b s i j', b=batch_size)

        # Apply all symmetry operations: rotB_sym = sym_matrices @ rotB
        rotB_sym = torch.einsum('bsik,bskj->bsij', sym_expanded, rotB_expanded)

        # Expand rotA for comparison with all symmetrized rotB
        # rotA_flat: (batch, 3, 3) -> (batch, n_sym, 3, 3)
        rotA_expanded = repeat(rotA_flat, 'b i j -> b s i j', s=n_sym)

        # Compute all traces at once: Tr(rotA @ rotB_sym^T)
        # This is equivalent to calling _compute_dot_trace for each pair
        rotB_sym_T = rotB_sym.transpose(-1, -2)  # (batch, n_sym, 3, 3)
        prod = torch.einsum('bsik,bskj->bsij', rotA_expanded, rotB_sym_T)
        traces = prod.diagonal(dim1=-1, dim2=-2).sum(-1)  # (batch, n_sym)

        # Convert traces to rotation errors
        rotation_errors = torch.arccos(torch.clamp((traces - 1) / 2, -1, 1))

        # Find minimum error across symmetry operations
        min_errors = torch.min(rotation_errors, dim=-1)[0]  # (batch,)

        # Reshape back to original batch shape
        if len(batch_shape) == 0:
            return min_errors.squeeze()
        elif len(batch_shape) == 1:
            return min_errors
        else:
            return min_errors.view(batch_shape)

def mean_rot_matrix(rotMatrices: torch.Tensor, dim:int, weights:Optional[torch.Tensor]=None, compute_dispersion:bool=True):

    assert 0 <= dim < len(rotMatrices.shape)-2
    if weights is None:
        weights = torch.ones_like(rotMatrices[:,:,:,0,0]).softmax(-1)
    else:
        weights = weights.to(rotMatrices).unsqueeze(-1).unsqueeze(-1)
    m = (weights * rotMatrices).sum(dim)
    U, S, Vh = torch.linalg.svd(m)
    avg = U @ Vh

    # Correct for possible reflection (det = -1)
    # Check determinant along the specified dimension, adjusting the last column of U if needed
    adjust = torch.linalg.det(avg) < 0  # A boolean tensor marking where the determinant is negative

    if adjust.any():
        # Expand adjust to match the dimensionality for broadcasting
        adjust = adjust.unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 3)
        # Correcting only the last column of U where needed
        U_adj = U.clone()
        U_adj[adjust][:, :, -1] *= -1
        avg = torch.matmul(U_adj, Vh)

    if compute_dispersion:
         disp = rotation_error_rads(avg.unsqueeze(dim), rotMatrices).mean(dim)
    else:
        disp = None
    return avg, disp

if __name__ == "__main__":
    from scipy.spatial.transform import Rotation
    ra = torch.FloatTensor(Rotation.random(32).as_matrix())
    rb = torch.FloatTensor(Rotation.random(32).as_matrix())
    out = rotation_error_rads(ra, rb)
    print(out)
    out2 = rotation_error_with_sym(ra, rb, symmetry="C1")
    print(out2)
    assert out.allclose(out2)
    sym = "D2"
    symMats = getSymmetryGroup(sym, as_matrix=True, device=ra.device).unsqueeze(0).expand(ra.shape[0], -1, -1, -1)[torch.arange(ra.shape[0]), torch.randint(0,4, size=(ra.shape[0],)),...]
    rc = symMats @ ra
    out3 = rotation_error_with_sym(ra, rc, symmetry=sym)
    print(out3)
    assert out3.allclose(torch.zeros_like(out3), atol=1e-3)