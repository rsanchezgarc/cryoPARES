from typing import Union, Tuple, Optional

import torch



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

    :param rot: tensor of shape (*,3,3)
    :return: The rotation magnitude in radians
    """
    trace = rot.diagonal(dim1=-1, dim2=-2).sum(-1)
    return torch.arccos(torch.clamp((trace - 1) / 2, -1, 1))

def nearest_rotmat_idx(src, targets):
    """

    :param src: tensor of shape (B, 3, 3)
    :param targets: tensor of shape (*, 3, 3)
    :return:
        - dot_trace: The value of the trace of the selected nearest rotmat idxs
        - idxs: The idxs of the nearest rotation matrices

    '''return index of target that is nearest to each element in src
    uses negative trace of the dot product to avoid arccos operation
    :src: tensor of shape (B, 3, 3)
    :target: tensor of shape (*, 3, 3)
    '''
    """
    trace = _compute_dot_trace(src.unsqueeze(1), targets.unsqueeze(0))
    dot_trace, idxs = torch.max(trace, dim=1)
    return dot_trace, idxs

#TODO: Implement the case that has symmetry for rotation_error_rads
def rotation_error_rads(rotA, rotB):
    """

    :param rotA: tensor of shape (*,3,3)
    :param rotB: tensor of shape (*,3,3)
    :return: rotation error in radians, tensor of shape (*)
    """

    trace = _compute_dot_trace(rotA, rotB)
    return torch.arccos(torch.clamp((trace - 1) / 2, -1, 1))



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