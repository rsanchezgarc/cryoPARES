import math
import torch


# =========================
# Helpers
# =========================

def _fibonacci_sphere(n: int, device=None, dtype=torch.float32, phi0: float = 0.0):
    """Deterministic, well-spread directions on S^2 (full sphere)."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    k  = torch.arange(n, device=device, dtype=dtype) + 0.5
    z  = 1 - 2*k/n
    r  = torch.sqrt(torch.clamp(1 - z*z, min=0))
    ga = math.pi * (3 - math.sqrt(5))  # golden angle
    phi = phi0 + ga * (k - 1)
    x, y = r*torch.cos(phi), r*torch.sin(phi)
    return torch.stack([x, y, z], dim=-1)  # (n,3)


def _skew_from_vec(v: torch.Tensor) -> torch.Tensor:
    """Return [v]_x for v (...,3)."""
    wx, wy, wz = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype, device=v.device)
    O[..., 0, 1], O[..., 0, 2] = -wz,  wy
    O[..., 1, 0], O[..., 1, 2] =  wz, -wx
    O[..., 2, 0], O[..., 2, 1] = -wy,  wx
    return O


def _exp_map_so3(v: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues using rotation vectors v = theta * u.
    R = I + sinc(theta) [v]_x + ((1 - cos theta) / theta^2) [v]_x^2
    with stable Taylor fallbacks near theta -> 0.
    """
    th = torch.linalg.norm(v, dim=-1, keepdim=True)
    eps = 1e-8
    A = torch.where(th > eps, torch.sin(th)/th, 1 - th**2/6 + th**4/120)
    B = torch.where(th > eps, (1 - torch.cos(th))/(th**2), 0.5 - th**2/24 + th**4/720)
    O = _skew_from_vec(v)
    I = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(O)
    return I + A[..., None] * O + B[..., None] * (O @ O)


def _matrix_to_zyz(R: torch.Tensor):
    """
    ZYZ extraction suitable near identity:
      beta  = arccos(R33)
      alpha = atan2(R23, R13)
      gamma = atan2(R32, -R31)
    """
    R13, R23, R33 = R[..., 0, 2], R[..., 1, 2], R[..., 2, 2]
    R31, R32      = R[..., 2, 0], R[..., 2, 1]
    beta  = torch.arccos(R33.clamp(-1.0, 1.0))
    alpha = torch.atan2(R23, R13)
    gamma = torch.atan2(R32, -R31)
    return alpha, beta, gamma


def _zyz_to_matrix(alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Build Rz(alpha) @ Ry(beta) @ Rz(gamma)."""
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta),  torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)

    Rz_a = torch.stack([
        torch.stack([ca, -sa, torch.zeros_like(ca)], dim=-1),
        torch.stack([sa,  ca, torch.zeros_like(ca)], dim=-1),
        torch.stack([torch.zeros_like(ca), torch.zeros_like(ca), torch.ones_like(ca)], dim=-1),
    ], dim=-2)

    Ry_b = torch.stack([
        torch.stack([ cb, torch.zeros_like(cb), sb], dim=-1),
        torch.stack([torch.zeros_like(cb), torch.ones_like(cb), torch.zeros_like(cb)], dim=-1),
        torch.stack([-sb, torch.zeros_like(cb), cb], dim=-1),
    ], dim=-2)

    Rz_g = torch.stack([
        torch.stack([cg, -sg, torch.zeros_like(cg)], dim=-1),
        torch.stack([sg,  cg, torch.zeros_like(cg)], dim=-1),
        torch.stack([torch.zeros_like(cg), torch.zeros_like(cg), torch.ones_like(cg)], dim=-1),
    ], dim=-2)

    return Rz_a @ Ry_b @ Rz_g


def _wrap_angle_diff(a: torch.Tensor) -> torch.Tensor:
    """Wrap radian differences to [-pi, pi] for robust comparison."""
    return (a + math.pi) % (2*math.pi) - math.pi


# -------------------------
# Gauss–Legendre (Golub–Welsch)
# -------------------------
def _gauss_legendre(n: int, a: float = -1.0, b: float = 1.0, device=None, dtype=torch.float32):
    """
    n-point Gauss–Legendre on [a,b].
    Returns nodes x (n,) and weights w (n,) as torch tensors.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    k = torch.arange(1, n, device=device, dtype=dtype)
    beta = k / torch.sqrt(4*k*k - 1)
    T = torch.zeros((n, n), device=device, dtype=dtype)
    T.diagonal(1).copy_(beta)
    T.diagonal(-1).copy_(beta)
    evals, evecs = torch.linalg.eigh(T)
    x = evals
    w = 2 * (evecs[0, :] ** 2)
    xm = (b - a) / 2 * x + (b + a) / 2
    wm = (b - a) / 2 * w
    return xm, wm


# =========================
# use_small_aprox near-identity sampler (tangent ω-ball)
# =========================

def so3_near_identity_by_spacing(
    distance_deg: float,
    spacing_deg:  float,
    device=None,
    dtype=torch.float32,
    return_weights: bool = True,
    output: str = "zyz_rad"
):
    """
    Deterministic near-identity sampler for SO(3) controlled by cap radius and target spacing (degrees).
    Returns rotations (angles or matrices) and Haar-normalized weights over the cap (if requested).
    use_small_aprox for *small* caps; γ effectively locked near β≈0.
    """
    if distance_deg <= 0 or spacing_deg <= 0:
        raise ValueError("distance_deg and spacing_deg must be > 0")
    if output not in ("zyz_rad", "zyz_deg", "matrix"):
        raise ValueError("output must be one of {'zyz_rad','zyz_deg','matrix'}")

    theta_max = math.radians(distance_deg)
    dtheta    = math.radians(spacing_deg)
    theta_max = max(theta_max, 1e-6)
    dtheta    = max(dtheta,    1e-6)

    # Number of radial shells so radial neighbor spacing ~ dtheta
    n_shells = max(1, math.ceil(theta_max / dtheta))

    # Equal-volume shells in ||ω|| with centers r_c and edges r_e
    j  = torch.arange(0, n_shells + 1, device=device, dtype=dtype)
    r_edges = theta_max * (j / n_shells) ** (1/3)                 # (S+1,)
    jc = torch.arange(n_shells, device=device, dtype=dtype)
    r_cent = theta_max * ((jc + 0.5) / n_shells) ** (1/3)         # (S,)

    omegas = []
    cell_vols = []

    # Precompute shell "cell volumes" in ω-space
    shell_vol = (4*math.pi/3) * (r_edges[1:]**3 - r_edges[:-1]**3)   # (S,)

    for s, r in enumerate(r_cent.tolist()):
        if r == 0.0:
            dirs = torch.tensor([[0., 0., 1.]], device=device, dtype=dtype)
            Ns = 1
            phi0 = 0.0
        else:
            # Tangential spacing near radius r is ~ r * sqrt(4π/N)
            # Set sqrt(4π/N) ~ dtheta / r  =>  N ~ 4π (r/dtheta)^2
            Ns_float = 4 * math.pi * (r / dtheta)**2
            Ns = max(8, int(math.ceil(Ns_float)))
            phi0 = (s * math.pi * (3 - math.sqrt(5))) % (2*math.pi)  # de-align shells
            dirs = _fibonacci_sphere(Ns, device=device, dtype=dtype, phi0=phi0)

        omega_s = r * dirs  # (Ns,3)
        omegas.append(omega_s)
        if return_weights:
            cv = (shell_vol[s] / Ns)
            cell_vols.append(torch.full((omega_s.shape[0],), cv.to(dtype), device=device))

    omega = torch.cat(omegas, dim=0)      # (N,3)
    R = _exp_map_so3(omega)               # (N,3,3)

    # Output
    if output == "matrix":
        rotations = R
    else:
        a, b, g = _matrix_to_zyz(R)
        angles = torch.stack([a, b, g], dim=-1)
        rotations = angles * (180/math.pi) if output == "zyz_deg" else angles

    if not return_weights:
        return rotations

    # Haar weights via exponential coordinates Jacobian J(theta) = (sin(theta/2)/theta)^2
    r = torch.linalg.norm(omega, dim=-1)
    J = torch.where(r > 0, (torch.sin(r/2) / (r + 1e-12))**2,
                    torch.tensor(0.25, device=r.device, dtype=r.dtype))  # limit at 0 = 1/4
    cell_vols = torch.cat(cell_vols, dim=0)
    w = cell_vols * J
    cap_vol = 4 * math.pi**2 * (1 - math.cos(theta_max))
    w = w * (cap_vol / (w.sum() + 1e-12))
    return rotations, w


# =========================
# Full-ring product quadrature (α×β×γ) over a β-cap
# =========================

def _so3_quadrature_cap_fullrings(distance_deg: float,
                                  spacing_deg: float,
                                  device=None,
                                  dtype=torch.float32,
                                  output="zyz_rad",
                                  return_weights=True):
    """
    Product quadrature with **full γ rings** at every (α,β):
      - α uniform in [0, 2π)
      - γ uniform in [0, 2π)
      - u = cosβ with Gauss–Legendre nodes on [cos β_max, 1]  (β ∈ [0, β_max])
    NOTE: The cap here is defined by **β ≤ β_max** (not geodesic θ). This guarantees full γ rings.
    For convolution this is safe; for cryo-EM matching you might prefer geodesic caps.
    """
    beta_max = math.radians(distance_deg)
    dtheta   = math.radians(spacing_deg)
    beta_max = max(beta_max, 1e-8)
    dtheta   = max(dtheta,   1e-8)

    # Heuristic node counts from spacing
    n_alpha = max(4, int(math.ceil(2*math.pi / dtheta)))
    n_gamma = n_alpha
    n_beta  = max(2, int(math.ceil(beta_max / dtheta)))

    # α, γ grids
    alpha = torch.linspace(0, 2*math.pi, steps=n_alpha+1, device=device, dtype=dtype)[:-1]
    gamma = torch.linspace(0, 2*math.pi, steps=n_gamma+1, device=device, dtype=dtype)[:-1]
    w_alpha = (2*math.pi) / n_alpha
    w_gamma = (2*math.pi) / n_gamma

    # Gauss–Legendre in u ∈ [cos β_max, 1], clamp for safety
    u_min = math.cos(beta_max)
    u_nodes, u_weights = _gauss_legendre(n_beta, a=u_min, b=1.0, device=device, dtype=dtype)
    u_nodes = u_nodes.clamp(min=u_min, max=1.0)
    beta = torch.arccos(u_nodes)  # (n_beta,)

    # Mesh
    A, B, G = torch.meshgrid(alpha, beta, gamma, indexing="ij")
    angles = torch.stack([A.flatten(), B.flatten(), G.flatten()], dim=-1)  # (N,3)

    if return_weights:
        # Product weights over (α, u, γ). Change of vars already handled by GL in u.
        w = (w_alpha * w_gamma) * u_weights.repeat(n_alpha * n_gamma)
        # Normalize to Haar volume of the β-cap: 4π^2 (1 - cos β_max)
        cap_vol = 4 * math.pi**2 * (1 - math.cos(beta_max))
        w = w * (cap_vol / (w.sum() + 1e-12))

    # Output
    if output == "zyz_rad":
        out = angles
    elif output == "zyz_deg":
        out = angles * (180.0 / math.pi)
    elif output == "matrix":
        out = _zyz_to_matrix(angles[:,0], angles[:,1], angles[:,2])
    else:
        raise ValueError("output must be 'zyz_rad','zyz_deg','matrix'")

    return (out, w) if return_weights else out


# =========================
# Unified API
# =========================

def so3_grid_near_identity_fibo(
    distance_deg: float,
    spacing_deg: float,
    use_small_aprox: bool = False,          # False: full γ rings (quadrature); True: tangent ω-ball (use_small_aprox near identity)
    device=None,
    dtype=torch.float32,
    return_weights: bool = True,
    output: str = "matrix"
):
    """
    Unified SO(3) grid:
      - use_small_aprox=False (default): full-ring product quadrature over a β-cap (safe for convolution).
      - use_small_aprox=True: near-identity ω-ball sampler (fast; γ effectively locked), good for tiny caps.

    All modes return Haar-normalized weights over the cap if return_weights=True.
    """
    if not use_small_aprox:
        return _so3_quadrature_cap_fullrings(distance_deg, spacing_deg,
                                             device=device, dtype=dtype,
                                             output=output, return_weights=return_weights)
    else:
        return so3_near_identity_by_spacing(distance_deg, spacing_deg,
                                            device=device, dtype=dtype,
                                            return_weights=return_weights, output=output)


# =========================
# Visualizers
# =========================

def plot_so3_distribution_3d_with_gamma(rots, probs=None, input_kind="matrix", ring_radius=0.03):
    """
    Visualize SO(3) samples on S^2 and make the in-plane angle γ visible:
    for each rotation, plot a small offset from the view direction whose azimuth encodes γ.
    Accepts batched shapes: (...,3,3) for matrices or (...,3) for ZYZ.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    def _ensure_matrix(R_or_angles, kind):
        if kind == "matrix":
            R = torch.as_tensor(R_or_angles)
        elif kind == "zyz_rad":
            a,b,g = torch.as_tensor(R_or_angles).unbind(-1)
            R = _zyz_to_matrix(a,b,g)
        elif kind == "zyz_deg":
            a,b,g = (torch.as_tensor(R_or_angles) * (math.pi/180.0)).unbind(-1)
            R = _zyz_to_matrix(a,b,g)
        else:
            raise ValueError("input_kind must be 'matrix' | 'zyz_rad' | 'zyz_deg'")
        return R  # (...,3,3)

    R = _ensure_matrix(rots, input_kind).detach().cpu().float().reshape(-1,3,3)
    # original γ if angles were provided; otherwise recover (ok for visualization)
    if input_kind.startswith("zyz"):
        gamma = torch.as_tensor(rots).reshape(-1,3)[:,2].detach().cpu().float()
    else:
        _,_,gamma = _matrix_to_zyz(R)
    # view dirs
    zhat = torch.tensor([0.,0.,1.], dtype=R.dtype)
    v = (R @ zhat).to(torch.float32)                 # (N,3)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)

    # local basis to paint γ ring
    tmp = torch.tensor([1.,0.,0.]).expand_as(v)
    swap = (v[:,0].abs() > 0.9)
    tmp[swap] = torch.tensor([0.,1.,0.])
    x = torch.linalg.cross(v, tmp); x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
    y = torch.linalg.cross(v, x)

    cosg = torch.cos(gamma).unsqueeze(-1)
    sing = torch.sin(gamma).unsqueeze(-1)
    pts = (v + ring_radius * (cosg * x + sing * y)).numpy()

    # sizes
    if probs is None:
        sizes = np.full(pts.shape[0], 18.0)
    else:
        w = torch.as_tensor(probs).detach().cpu().float()
        w = w.reshape(-1)
        w = w / (w.sum() + 1e-12)
        sizes = (36.0 * (w / (w.max() + 1e-12)).numpy()).clip(4, 64)

    # plot
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.hsv(0.5 + (gamma.detach().cpu().numpy() / (2*np.pi)))
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors, s=sizes, alpha=0.9)

    # sphere wireframe
    uu = np.linspace(0, 2*np.pi, 36); vv = np.linspace(0, np.pi, 18)
    xs = np.outer(np.cos(uu), np.sin(vv))
    ys = np.outer(np.sin(uu), np.sin(vv))
    zs = np.outer(np.ones_like(uu), np.cos(vv))
    ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.25)

    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.show()


def plot_rotation_vectors_shells(R):
    """Show ω = log(R) colored by ||ω|| to reveal shells (tangent mode)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    R = torch.as_tensor(R).detach()
    tr = torch.einsum("nii->n", R)
    theta = torch.arccos(((tr - 1) / 2).clamp(-1, 1))
    denom = (2*torch.sin(theta)).clamp_min(1e-12)
    v = torch.stack([
        R[:,2,1]-R[:,1,2],
        R[:,0,2]-R[:,2,0],
        R[:,1,0]-R[:,0,1]
    ], dim=1) / denom.unsqueeze(-1)
    omega = theta.unsqueeze(-1) * v
    r = omega.norm(dim=1)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(omega[:,0], omega[:,1], omega[:,2],
                   c=r.numpy(), s=10, cmap="viridis", alpha=0.9)
    fig.colorbar(p, ax=ax, shrink=0.6, label="‖ω‖ (rad)")
    ax.set_xlabel("ωx"); ax.set_ylabel("ωy"); ax.set_zlabel("ωz")
    ax.set_box_aspect([1,1,1])
    mx = float(omega.abs().max()) * 1.05
    ax.set_xlim(-mx, mx); ax.set_ylim(-mx, mx); ax.set_zlim(-mx, mx)
    plt.show()


# =========================
# Tests
# =========================

def _cap_volume(theta_max_rad: float) -> float:
    return 4 * math.pi**2 * (1 - math.cos(theta_max_rad))

def _test_quadrature_beta_cap_bounds_and_full_gamma():
    """Quadrature uses a β-cap: check max β and that each (α,β) has a full γ ring."""
    dist, sp = 18.0, 3.0
    ang, w = so3_grid_near_identity_fibo(dist, sp, use_small_aprox=False, return_weights=True, output="zyz_rad")
    beta = ang[:,1]
    assert float(beta.max()) <= math.radians(dist) + 1e-10, "β exceeds β_max in quadrature mode"

    # Count unique (α,β) cells and γ multiplicity
    a = ang[:,0].round(decimals=6)
    b = ang[:,1].round(decimals=6)
    g = ang[:,2].round(decimals=6)
    ab = torch.stack([a,b], dim=1)
    # map each (α,β) to multiplicity of γ
    uniq, inv = torch.unique(ab, dim=0, return_inverse=True)
    counts = torch.bincount(inv)
    # expect all counts equal to number of unique γ values
    n_gamma = torch.unique(g).numel()
    assert int(counts.min()) == int(counts.max()) == int(n_gamma), "Not full γ rings at each (α,β)"
    # weights sum check
    cap_vol = _cap_volume(math.radians(dist))
    rel_err = abs(float(w.sum()) - cap_vol) / cap_vol
    assert rel_err < 1e-6, f"Quadrature weights do not sum to cap volume (rel err {rel_err})"
    print("[OK] _test_quadrature_beta_cap_bounds_and_full_gamma")

def _test_tangent_geodesic_bounds():
    """use_small_aprox tangent sampler should respect geodesic cap θ ≤ distance."""
    dist, sp = 6.0, 1.0
    R, w = so3_grid_near_identity_fibo(dist, sp, use_small_aprox=True, return_weights=True, output="matrix")
    tr = torch.einsum("nii->n", R)
    theta = torch.acos(((tr - 1.0)/2.0).clamp(-1.0, 1.0))
    assert float(theta.max()) <= math.radians(dist) + 1e-6, "Tangent samples exceed geodesic cap"
    cap_vol = _cap_volume(math.radians(dist))
    rel_err = abs(float(w.sum()) - cap_vol) / cap_vol
    assert rel_err < 1e-6, f"Tangent weights do not sum to cap volume (rel err {rel_err})"
    print("[OK] _test_tangent_geodesic_bounds")

def _test_output_modes_and_roundtrips():
    ang, _ = so3_grid_near_identity_fibo(10.0, 2.0, use_small_aprox=False, return_weights=True, output="zyz_rad")
    R1 = _zyz_to_matrix(ang[:,0], ang[:,1], ang[:,2])
    a2,b2,g2 = _matrix_to_zyz(R1)
    R2 = _zyz_to_matrix(a2,b2,g2)
    diff = torch.linalg.norm(R1 - R2, dim=(-2,-1)).max().item()
    I = torch.eye(3).to(R1)
    ortho = torch.linalg.norm(R1.transpose(-1,-2) @ R1 - I, dim=(-2,-1)).max().item()
    deterr = (torch.det(R1) - 1.0).abs().max().item()
    assert diff < 1e-6 and ortho < 1e-6 and deterr < 1e-6, "Roundtrip / rotation tests failed"
    print("[OK] _test_output_modes_and_roundtrips")

def _test_determinism_and_scaling():
    a1,_ = so3_grid_near_identity_fibo(12.0, 1.0, use_small_aprox=False, return_weights=True, output="zyz_rad")
    a2,_ = so3_grid_near_identity_fibo(12.0, 1.0, use_small_aprox=False, return_weights=True, output="zyz_rad")
    assert torch.allclose(a1,a2), "Quadrature must be deterministic"
    # scaling: coarser spacing → fewer samples
    s_dense,_ = so3_grid_near_identity_fibo(12.0, 0.6, use_small_aprox=False, return_weights=True, output="zyz_rad")
    s_sparse,_ = so3_grid_near_identity_fibo(12.0, 1.2, use_small_aprox=False, return_weights=True, output="zyz_rad")
    assert s_sparse.shape[0] < s_dense.shape[0], "Coarser spacing should reduce samples"
    print("[OK] _test_determinism_and_scaling")


# =========================
# Demo in __main__
# =========================

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)

    # Tests
    _test_quadrature_beta_cap_bounds_and_full_gamma()
    _test_tangent_geodesic_bounds()
    _test_output_modes_and_roundtrips()
    _test_determinism_and_scaling()
    print("All tests passed.")

    # === Example usage ===
    # Full γ rings (safe, general)
    R_quad, w_quad = so3_grid_near_identity_fibo(30.0, 5.0, use_small_aprox=False, return_weights=True, output="matrix")
    # use_small_aprox alternative (tiny caps)
    R_tan,  w_tan  = so3_grid_near_identity_fibo(5.0, 1.0, use_small_aprox=True, return_weights=True, output="matrix")

    # === Visual checks (uncomment if you want to view) ===
    plot_so3_distribution_3d_with_gamma(R_quad, probs=w_quad, input_kind="matrix", ring_radius=0.04)
    plot_so3_distribution_3d_with_gamma(R_tan,  probs=w_tan,  input_kind="matrix", ring_radius=0.04)
    plot_rotation_vectors_shells(R_tan)
