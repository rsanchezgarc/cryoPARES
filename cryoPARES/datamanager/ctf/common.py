import torch


def _compute_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None):
    '''
    Compute the 2D CTF

    Input:
        freqs (Tensor) Nx2 or BxNx2 tensor of 2D spatial frequencies
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    '''
    assert freqs.shape[-1] == 2
    # convert units
    volt = volt * 1000
    cs = cs * 10 ** 7
    dfang = dfang * torch.pi / 180
    phase_shift = phase_shift * torch.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / (volt + 0.97845e-6 * volt ** 2) ** .5
    x = freqs[..., 0]
    y = freqs[..., 1]
    ang = torch.atan2(y, x)
    s2 = x ** 2 + y ** 2
    df = .5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
    gamma = 2 * torch.pi * (-.5 * df * lam * s2 + .25 * cs * lam ** 3 * s2 ** 2) - phase_shift
    ctf = (1 - w ** 2) ** .5 * torch.sin(gamma) - w * torch.cos(gamma)
    if bfactor is not None:
        ctf *= torch.exp(-bfactor / 4 * s2)
    return -ctf


def convert_fft_to_rfft(ctf_fft):
    """
    Convert a FFT CTF to RFFT format, handling both odd and even sizes properly.

    For even N:
        rfft gives N//2 + 1 frequencies [0, 1, ..., N//2]
    For odd N:
        rfft gives (N+1)//2 frequencies [0, 1, ..., (N-1)//2]
    """
    N = ctf_fft.shape[-1]
    centre = N // 2

    if N % 2 == 0:  # Even size
        ctf_rfft = torch.empty(*ctf_fft.shape[:-1], N // 2 + 1,
                               dtype=ctf_fft.dtype, device=ctf_fft.device)
        ctf_rfft[..., :-1] = ctf_fft[..., centre:]  # [centre:N]
        ctf_rfft[..., -1] = ctf_fft[..., 0]  # Nyquist frequency
    else:  # Odd size
        ctf_rfft = torch.empty(*ctf_fft.shape[:-1], (N + 1) // 2,
                               dtype=ctf_fft.dtype, device=ctf_fft.device)
        ctf_rfft = ctf_fft[..., centre:]  # [centre:N]

    return ctf_rfft
