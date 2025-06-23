from typing import Tuple

import torch


_OFFSETS = torch.tensor(
    [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],
    dtype=torch.long
)

def _insert_linear_3d_tensor_scatter(
    data: torch.Tensor,        # (b, c)
    coordinates: torch.Tensor, # (b, 3)  float32 / float64
    image: torch.Tensor,       # (c, D, H, W)
    weights: torch.Tensor,     # (D, H, W)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This is to be used for cryo-EM reconstruction. Typically image is a 3 channel image with the real and imaginary parts
    of the volume, and the third channel is the ctf**2. We use real and imaginary parts separatelly, because torch is better
    optimized for that.

    """
    device           = image.device
    dtype            = data.dtype
    b, c             = data.shape
    _, D, H, W       = image.shape
    N                = D * H * W

    offs             = _OFFSETS.to(device)          # (8,3)

    # ---- integer base corner & fractional part --------------------------------
    base_int         = torch.floor(coordinates).to(torch.long)      # (b,3)
    frac             = (coordinates - base_int).to(dtype)           # (b,3)

    # ---- eight voxel indices (b,8,3) ------------------------------------------
    zyx8             = base_int[:, None, :] + offs                   # broadcast add

    # *** in-place clamp – fixes out-of-bounds completely ***
    zyx8[..., 0].clamp_(0, D - 1)   # z
    zyx8[..., 1].clamp_(0, H - 1)   # y
    zyx8[..., 2].clamp_(0, W - 1)   # x

    # ---- weights for the 8 corners (b,8) --------------------------------------
    tz, ty, tx       = frac.T                                        # each (b,)
    w_corner         = torch.stack((
        (1-tz)*(1-ty)*(1-tx),
        (1-tz)*(1-ty)*tx,
        (1-tz)*ty*(1-tx),
        (1-tz)*ty*tx,
        tz*(1-ty)*(1-tx),
        tz*(1-ty)*tx,
        tz*ty*(1-tx),
        tz*ty*tx,
    ), dim=1).to(dtype)                                             # (b,8)

    # ---- flat indices ----------------------------------------------------------
    flat_idx         = ((zyx8[...,0]*H + zyx8[...,1]) * W + zyx8[...,2]).view(-1)

    # ---- prepare channel-wise contributions -----------------------------------
    contrib          = (data.unsqueeze(2) * w_corner.unsqueeze(1))   # (b,c,8)
    contrib          = contrib.permute(1,0,2).contiguous().view(c, -1)

    # ---- scatter into image ----------------------------------------------------
    img_flat         = image.view(c, -1)
    idx_expand       = flat_idx.unsqueeze(0).expand(c, -1)          # (c, b*8)
    img_flat.scatter_add_(1, idx_expand, contrib)

    # ---- scatter into weights --------------------------------------------------
    w_flat           = weights.view(-1)
    w_flat.scatter_add_(0, flat_idx, w_corner.view(-1))

    return image, weights

