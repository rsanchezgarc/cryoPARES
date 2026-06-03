"""
Gold-standard FSC utilities for cryo-EM post-processing.

Implements phase-randomisation mask-bias correction following the
Chen et al. (2013) procedure as used in RELION.
"""
import numpy as np
import torch

from cryoPARES.scripts.computeFsc import compute_fsc, first_crossing_with_bounce_check, _torch_shell_sums
from cryoPARES.postprocessing.sharpening import _get_device


def compute_shell_amplitudes(vol_np: np.ndarray, px_A: float, device=None):
    """
    Compute radial-mean amplitude (|FFT|) per Fourier shell.

    Returns
    -------
    amplitudes : 1D array, shape (D//2 - 1,)
        Mean |FFT| in each shell.
    spatial_freq : 1D array, shape (D//2 - 1,)
        Spatial frequency axis in 1/Å for each shell.
    """
    dev = _get_device(device)
    D = vol_np.shape[0]
    H, W = vol_np.shape[1], vol_np.shape[2]
    num_shells = D // 2

    vol_t = torch.as_tensor(vol_np.astype(np.float32), device=dev)

    # rfftn: shape (D, H, W//2+1); for per-shell mean amplitude the
    # missing conjugate half cancels (numerator and denominator both halved).
    ft = torch.fft.rfftn(vol_t)
    amp = ft.abs()

    # Broadcast 1-D coords instead of meshgrid (saves 2×430 MB for 476³).
    kz = (torch.fft.fftfreq(D,  device=dev) * D).reshape(-1, 1, 1)
    ky = (torch.fft.fftfreq(H,  device=dev) * H).reshape(1, -1, 1)
    kx = (torch.fft.rfftfreq(W, device=dev) * W).reshape(1, 1, -1)
    k_dist = torch.sqrt(kz**2 + ky**2 + kx**2)

    bin_edges_t  = torch.arange(0.5, num_shells, 1.0, dtype=torch.float32, device=dev)
    shell_radii  = ((bin_edges_t[:-1] + bin_edges_t[1:]) / 2).cpu().numpy()

    amp_f = amp.reshape(-1)
    k_f   = k_dist.reshape(-1)
    idx = torch.bucketize(k_f, bin_edges_t) - 1
    valid = (idx >= 0) & (idx < (num_shells - 1))
    idx_v = idx[valid].long()
    n_bins = num_shells - 1

    amp_sum = torch.zeros(n_bins, dtype=torch.float32, device=dev).scatter_add_(0, idx_v, amp_f[valid])
    counts  = torch.zeros(n_bins, dtype=torch.float32, device=dev).scatter_add_(0, idx_v,
                  torch.ones(valid.sum(), dtype=torch.float32, device=dev))
    counts  = counts.clamp(min=1)
    amplitudes = (amp_sum / counts).cpu().numpy()

    spatial_freq = shell_radii / (D * px_A)
    return amplitudes.astype(np.float32), spatial_freq.astype(np.float32)


def randomize_phases_beyond(vol_np: np.ndarray, shell_radius_pix: float,
                             rng: np.random.Generator = None,
                             device=None) -> np.ndarray:
    """
    Return a new volume with the same amplitudes as *vol_np* but with
    uniformly random phases for all Fourier voxels whose distance from the
    DC component is >= *shell_radius_pix*.

    Parameters
    ----------
    vol_np : np.ndarray  (D, H, W)
    shell_radius_pix : float
        Randomize phases at shells with k_dist >= this value (pixels).
    rng : numpy Generator, optional
    device : torch.device or None — GPU if available, else CPU

    Returns
    -------
    np.ndarray (D, H, W), dtype float32, real-space
    """
    import math
    if rng is None:
        rng = np.random.default_rng()

    dev = _get_device(device)
    D, H, W = vol_np.shape

    vol_t = torch.as_tensor(vol_np.astype(np.float32), device=dev)

    # Full complex FFT; DC at [0,0,0] (no fftshift, saves an 861 MB copy).
    # k_dist uses fftfreq*N so distances are identical to the fftshifted version.
    ft = torch.fft.fftn(vol_t)  # complex64, DC at [0,0,0]

    # Broadcast 1-D coordinate vectors instead of meshgrid (saves 2×430 MB).
    kz = (torch.fft.fftfreq(D, device=dev) * D).reshape(-1, 1, 1)
    ky = (torch.fft.fftfreq(H, device=dev) * H).reshape(1, -1, 1)
    kx = (torch.fft.fftfreq(W, device=dev) * W).reshape(1, 1, -1)
    k_dist = torch.sqrt(kz**2 + ky**2 + kx**2)

    rand_mask = k_dist >= shell_radius_pix
    # Free coord/distance tensors before large allocations
    del k_dist, kz, ky, kx
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    amp = ft[rand_mask].abs()  # float32, shape (N_rand,)

    # Generate random phases with numpy RNG for reproducibility, then move to device
    n_rand = int(rand_mask.sum().item())
    phase_np = rng.uniform(0, 2 * math.pi, size=n_rand).astype(np.float32)
    phase_t = torch.as_tensor(phase_np, device=dev)
    del phase_np

    # torch.polar(abs, angle) = abs * exp(1j*angle): avoids intermediate complex tensor
    ft[rand_mask] = torch.polar(amp, phase_t)
    del amp, phase_t

    result = torch.fft.ifftn(ft).real.to(torch.float32)
    return result.cpu().numpy()


def find_randomization_shell(fsc_unmasked: np.ndarray,
                              spatial_freq: np.ndarray,
                              threshold: float = 0.8) -> tuple:
    """
    Return (shell_radius_pix, shell_idx) for the first shell where the
    unmasked FSC drops below *threshold*.

    Parameters
    ----------
    fsc_unmasked : 1D array
    spatial_freq : 1D array (1/Å)
    threshold : float

    Returns
    -------
    shell_radius_pix : float   — the radial distance in pixels corresponding
                                  to the chosen shell (approx shell_idx + 0.5)
    shell_idx : int            — index into fsc_unmasked / spatial_freq
    """
    # The FSC shells go from lower to higher resolution (higher spatial freq).
    # We search for the first crossing from high to low FSC.
    f = np.asarray(fsc_unmasked)
    below = np.where(f < threshold)[0]
    if below.size == 0:
        # FSC never drops below threshold — randomize only Nyquist region
        shell_idx = len(f) - 1
    else:
        shell_idx = int(below[0])

    # shell_radius_pix: index i → radii in [i+0.5, i+1.5), center ≈ i+0.5
    # We use i+0.5 as the cutoff for phase randomization, which is the
    # lower edge of the shell (conservative: randomize from that shell onward).
    shell_radius_pix = float(shell_idx) + 0.5
    return shell_radius_pix, shell_idx


def compute_corrected_fsc(fsc_masked: np.ndarray,
                           fsc_random_masked: np.ndarray,
                           rand_shell_idx: int) -> np.ndarray:
    """
    Phase-randomisation corrected FSC (Chen et al. 2013).

    For shells **below** rand_shell_idx the phases were NOT randomized, so
    FSC_random_masked ≈ FSC_masked there and the correction formula would
    incorrectly return ≈0.  We therefore use FSC_masked directly for those
    low-resolution shells.

    For shells **at and beyond** rand_shell_idx, signal is negligible and the
    mask inflates the FSC; the correction removes that bias:
        FSC_true = (FSC_masked - FSC_rand) / (1 - FSC_rand)

    Result clipped to [-0.1, 1.0].
    """
    eps = 1e-6
    corrected = np.copy(fsc_masked).astype(np.float32)

    if rand_shell_idx < len(fsc_masked):
        hi = slice(rand_shell_idx, len(fsc_masked))
        num   = fsc_masked[hi] - fsc_random_masked[hi]
        denom = 1.0 - fsc_random_masked[hi] + eps
        corrected[hi] = (num / denom).astype(np.float32)

    return np.clip(corrected, -0.1, 1.0).astype(np.float32)


def _subtract_background(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Subtract the mean density of the soft-edge border region from *vol*,
    then apply the mask.  Matches RELION's softMaskOutsideMap step.

    The border is defined as voxels where 0 < mask < 1 (the soft-edge
    transition zone).  If no such voxels exist the global mean outside
    the mask (mask == 0) is used as fallback; if the mask is binary
    (all 0 or 1) there is no border, so the mean of the unmasked region
    is subtracted instead.
    """
    border = (mask > 0.0) & (mask < 1.0)
    if border.any():
        bg = float(vol[border].mean())
    else:
        outside = mask == 0.0
        bg = float(vol[outside].mean()) if outside.any() else 0.0
    return ((vol - bg) * mask).astype(np.float32)


def run_gold_standard_fsc(half1: np.ndarray, half2: np.ndarray,
                           mask: np.ndarray, px_A: float, device=None) -> dict:
    """
    Full gold-standard FSC pipeline with phase-randomisation correction.

    Parameters
    ----------
    half1, half2 : np.ndarray (D, H, W) float32
    mask : np.ndarray (D, H, W) float32 — values in [0, 1]
    px_A : float — pixel size in Å

    Returns
    -------
    dict with keys:
        fsc_unmasked, fsc_masked, fsc_random_masked, fsc_corrected,
        spatial_freq, resolution_A, res_A_0143, res_A_05
    """
    import torch as _torch

    def _cache_clear():
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    # 1. Unmasked FSC
    fsc_unmasked, spatial_freq, resolution_A, (res_05_unmasked, _) = compute_fsc(
        half1, half2, px_A, mask=None, device=device)
    _cache_clear()

    # 2. Find randomization shell
    rand_shell_pix, rand_shell_idx = find_randomization_shell(fsc_unmasked, spatial_freq)
    rand_res_A = resolution_A[rand_shell_idx] if rand_shell_idx < len(resolution_A) else float("nan")
    print(f"Phase randomization shell: idx={rand_shell_idx}, "
          f"radius={rand_shell_pix:.1f} px, res≈{rand_res_A:.1f} Å")

    # 3. Masked FSC — subtract background (border mean) before masking,
    #    matching RELION's softMaskOutsideMap step.
    h1_masked = _subtract_background(half1, mask)
    h2_masked = _subtract_background(half2, mask)
    fsc_masked, _, _, _ = compute_fsc(h1_masked, h2_masked, px_A, mask=None, device=device)
    _cache_clear()

    # 4. Randomize phases of both half-maps beyond randomization shell,
    #    then compute FSC with mask applied (same background subtraction).
    rng = np.random.default_rng(seed=0)
    h1_rand = randomize_phases_beyond(half1, rand_shell_pix, rng=rng, device=device)
    _cache_clear()
    h2_rand = randomize_phases_beyond(half2, rand_shell_pix, rng=rng, device=device)
    _cache_clear()
    h1_rand_masked = _subtract_background(h1_rand, mask)
    h2_rand_masked = _subtract_background(h2_rand, mask)
    del h1_rand, h2_rand
    fsc_random_masked, _, _, _ = compute_fsc(h1_rand_masked, h2_rand_masked, px_A, mask=None, device=device)
    _cache_clear()

    # 5. Corrected FSC — only apply formula at shells >= randomization shell
    fsc_corrected = compute_corrected_fsc(fsc_masked, fsc_random_masked, rand_shell_idx)

    # 6. Resolution estimates from corrected FSC
    res_A_0143, _ = first_crossing_with_bounce_check(
        fsc_corrected, resolution_A,
        threshold=0.143, cutoff_res_A=15.0)
    res_A_05, _ = first_crossing_with_bounce_check(
        fsc_corrected, resolution_A,
        threshold=0.5, cutoff_res_A=15.0)

    return {
        "fsc_unmasked":      fsc_unmasked,
        "fsc_masked":        fsc_masked,
        "fsc_random_masked": fsc_random_masked,
        "fsc_corrected":     fsc_corrected,
        "spatial_freq":      spatial_freq,
        "resolution_A":      resolution_A,
        "res_A_0143":        res_A_0143,
        "res_A_05":          res_A_05,
        "rand_shell_pix":    rand_shell_pix,
    }
