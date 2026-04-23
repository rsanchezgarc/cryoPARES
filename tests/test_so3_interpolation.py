"""
Minimal tests for Change #7 — SO(3) sub-step Euler interpolation.

Run with:
    python -m pytest tests/test_so3_interpolation.py -v
or directly:
    python tests/test_so3_interpolation.py
"""
import math
import sys
import os
import tempfile

import torch

# ── make sure the project is importable ────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cryoPARES.projmatching.projMatcher import (
    _precompute_so3_interp_neighbors,
    _so3_interpolate_euler_winner,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Unit tests for _precompute_so3_interp_neighbors
# ═══════════════════════════════════════════════════════════════════════════

def test_neighbor_table_shape():
    """6°/2° grid → 7³ = 343 points, 6 neighbor columns."""
    nb_idx, nb_valid = _precompute_so3_interp_neighbors(6.0, 2.0)
    assert nb_idx.shape == (343, 6),   f"wrong shape: {nb_idx.shape}"
    assert nb_valid.shape == (343, 6), f"wrong shape: {nb_valid.shape}"
    print("  [OK] neighbor table shape 343×6")


def test_corner_has_three_valid_neighbors():
    """Index 0 = (rot=0, tilt=0, psi=0): only the three + neighbors are valid."""
    nb_idx, nb_valid = _precompute_so3_interp_neighbors(6.0, 2.0)
    # columns: rot+, rot-, tilt+, tilt-, psi+, psi-
    expected = [True, False, True, False, True, False]
    assert nb_valid[0].tolist() == expected, (
        f"corner valid mask wrong: {nb_valid[0].tolist()}"
    )
    print("  [OK] corner (0,0,0) has exactly 3 valid neighbors")


def test_center_has_all_six_neighbors():
    """Center point (3,3,3) on a 7×7×7 grid: all 6 neighbors valid."""
    nb_idx, nb_valid = _precompute_so3_interp_neighbors(6.0, 2.0)
    n = 7
    center = 3 * n * n + 3 * n + 3
    assert nb_valid[center].all(), (
        f"center valid mask wrong: {nb_valid[center].tolist()}"
    )
    print("  [OK] center (3,3,3) has all 6 valid neighbors")


def test_neighbor_indices_correct():
    """Spot-check: center's rot+ neighbor = center + 49, tilt+ = center + 7, psi+ = center + 1."""
    nb_idx, nb_valid = _precompute_so3_interp_neighbors(6.0, 2.0)
    n = 7
    center = 3 * n * n + 3 * n + 3
    assert nb_idx[center, 0] == center + 49, "rot+ wrong"   # col 0 = rot+
    assert nb_idx[center, 2] == center + 7,  "tilt+ wrong"  # col 2 = tilt+
    assert nb_idx[center, 4] == center + 1,  "psi+ wrong"   # col 4 = psi+
    assert nb_idx[center, 1] == center - 49, "rot- wrong"
    assert nb_idx[center, 3] == center - 7,  "tilt- wrong"
    assert nb_idx[center, 5] == center - 1,  "psi- wrong"
    print("  [OK] neighbor flat indices are correct for center point")


def test_boundary_indices_clamped_to_self():
    """Invalid neighbors must be clamped to the point itself (guard against OOB access)."""
    nb_idx, nb_valid = _precompute_so3_interp_neighbors(6.0, 2.0)
    # Index 0: rot-, tilt-, psi- are invalid → should point back to 0
    assert nb_idx[0, 1] == 0, "rot- of corner should be clamped to 0"
    assert nb_idx[0, 3] == 0, "tilt- of corner should be clamped to 0"
    assert nb_idx[0, 5] == 0, "psi- of corner should be clamped to 0"
    print("  [OK] boundary indices correctly clamped to self")


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Unit tests for _so3_interpolate_euler_winner
# ═══════════════════════════════════════════════════════════════════════════

def test_interpolation_output_shape():
    nb_idx, nb_valid = _precompute_so3_interp_neighbors(6.0, 2.0)
    B, nDelta = 8, 343
    perImgCorr = torch.randn(B, nDelta)
    winner_idx = torch.randint(0, nDelta, (B,))
    delta = _so3_interpolate_euler_winner(perImgCorr, winner_idx, nb_idx, nb_valid, 2.0)
    assert delta.shape == (B, 3), f"wrong shape: {delta.shape}"
    print("  [OK] output shape (B, 3)")


def test_interpolation_bounded_by_half_step():
    """Corrections must stay within ±step/2 = ±1°."""
    nb_idx, nb_valid = _precompute_so3_interp_neighbors(6.0, 2.0)
    B, nDelta, step = 32, 343, 2.0
    torch.manual_seed(0)
    perImgCorr = torch.randn(B, nDelta)
    winner_idx = torch.randint(1, nDelta - 1, (B,))  # avoid boundary points
    delta = _so3_interpolate_euler_winner(perImgCorr, winner_idx, nb_idx, nb_valid, step)
    assert (delta.abs() <= step / 2 + 1e-6).all(), (
        f"delta exceeds ±{step/2}°: max={delta.abs().max():.4f}°"
    )
    print(f"  [OK] all corrections within ±{step/2}° (max seen: {delta.abs().max():.4f}°)")


def test_boundary_winner_returns_zero_on_invalid_axis():
    """For a corner point, the axes pointing outside the grid must give delta=0."""
    nb_idx, nb_valid = _precompute_so3_interp_neighbors(6.0, 2.0)
    B, nDelta, step = 4, 343, 2.0
    # Put the winner at index 0 (corner: rot-, tilt-, psi- all invalid)
    perImgCorr = torch.randn(B, nDelta)
    winner_idx = torch.zeros(B, dtype=torch.long)
    delta = _so3_interpolate_euler_winner(perImgCorr, winner_idx, nb_idx, nb_valid, step)
    # All axes at the corner have at least one missing neighbor → all deltas must be 0
    assert (delta == 0).all(), f"corner should give zero correction: {delta}"
    print("  [OK] corner winner returns zero correction on all axes")


def test_parabolic_peak_at_winner():
    """When the CC surface is a perfect parabola with peak at the winner, delta should be ~0."""
    nb_idx, nb_valid = _precompute_so3_interp_neighbors(6.0, 2.0)
    n, step = 7, 2.0
    nDelta = n ** 3
    B = 1

    # Center index (3,3,3)
    center = 3 * n * n + 3 * n + 3
    winner_idx = torch.tensor([center])

    # Build a CC array that's a perfect downward parabola centered on 'center'
    # CC(k) = C - |k - center|²  → peak exactly at center → delta should be 0
    flat = torch.arange(nDelta, dtype=torch.float)
    rot_i  = flat // (n * n)
    tilt_j = (flat // n) % n
    psi_l  = flat % n
    perImgCorr = (10.0
                  - (rot_i  - 3).pow(2)
                  - (tilt_j - 3).pow(2)
                  - (psi_l  - 3).pow(2)).unsqueeze(0)  # (1, 343)

    delta = _so3_interpolate_euler_winner(perImgCorr, winner_idx, nb_idx, nb_valid, step)
    assert (delta.abs() < 1e-4).all(), (
        f"perfect parabola at winner should give ~0 delta: {delta}"
    )
    print("  [OK] perfect parabola at winner gives near-zero correction")


def test_parabolic_peak_offset_from_winner():
    """When the true peak is half a step ahead, the correction should be ~+step/2."""
    nb_idx, nb_valid = _precompute_so3_interp_neighbors(6.0, 2.0)
    n, step = 7, 2.0
    nDelta = n ** 3
    B = 1

    # Winner at center; true continuous peak is shifted by +0.5 step along rot axis
    # A 1D parabola: CC(k) = -(k - (3.5))² + const  →  evaluated at integers 3, 4 (and 2)
    # CC(2) = -(2-3.5)² = -2.25
    # CC(3) = -(3-3.5)² = -0.25   ← winner (integer argmax)
    # CC(4) = -(4-3.5)² = -0.25   ← tie at winner+1
    # Parabolic interpolation: delta_rot = (CC(2) - CC(4)) / (2*CC(2) - 4*CC(3) + 2*CC(4)) * step
    #   = (-2.25 - (-0.25)) / (2*(-2.25) - 4*(-0.25) + 2*(-0.25)) * 2
    #   = (-2.0) / (-4.5 + 1.0 - 0.5) * 2 = (-2.0) / (-4.0) * 2 = 1.0° = step/2  ✓

    perImgCorr = torch.zeros(B, nDelta)
    center = 3 * n * n + 3 * n + 3
    perImgCorr[0, center - 49] = -2.25   # rot-1 (index center - stride_rot)
    perImgCorr[0, center]      = -0.25   # rot=3  (winner)
    perImgCorr[0, center + 49] = -0.25   # rot+1

    winner_idx = torch.tensor([center])
    delta = _so3_interpolate_euler_winner(perImgCorr, winner_idx, nb_idx, nb_valid, step)

    # rot correction should be +1.0° (= step/2); tilt and psi should be 0
    assert abs(delta[0, 0].item() - 1.0) < 1e-4, f"rot delta wrong: {delta[0, 0].item()}"
    assert abs(delta[0, 1].item()) < 1e-4, f"tilt delta should be 0: {delta[0, 1].item()}"
    assert abs(delta[0, 2].item()) < 1e-4, f"psi delta should be 0: {delta[0, 2].item()}"
    print(f"  [OK] offset parabola gives correct delta: {delta[0].tolist()}")


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Integration: ProjectionMatcher init + tiny GPU forward pass
# ═══════════════════════════════════════════════════════════════════════════

def test_projmatcher_init_with_flag(gpu_id=1):
    """ProjectionMatcher should precompute neighbor table when use_so3_interpolation=True."""
    from cryoPARES.configs.mainConfig import main_config
    main_config.projmatching.use_so3_interpolation = True

    try:
        from cryoPARES.projmatching.projMatcher import ProjectionMatcher
        REF = "/home/rsanchez/cryo/data/EMPIAR-download/EMPIAR-10166/data/reconstruct.mrc"
        pm = ProjectionMatcher(reference_vol=REF)

        assert hasattr(pm, "_so3_interp_nb_idx"), "neighbor table not set on instance"
        assert pm._so3_interp_nb_idx.shape == (343, 6), pm._so3_interp_nb_idx.shape
        assert pm.use_so3_interpolation is True
        print("  [OK] ProjectionMatcher init sets neighbor table correctly")
        return pm
    finally:
        main_config.projmatching.use_so3_interpolation = False


def test_tiny_gpu_projmatching(gpu_id=1):
    """End-to-end projmatching with n=5 particles on GPU, use_so3_interpolation=True."""
    import starfile
    from cryoPARES.configs.mainConfig import main_config
    main_config.projmatching.use_so3_interpolation = True

    try:
        from cryoPARES.projmatching.projMatcher import ProjectionMatcher

        REF  = "/home/rsanchez/cryo/data/EMPIAR-download/EMPIAR-10166/data/reconstruct.mrc"
        STAR = "/home/rsanchez/cryo/data/cryopares/tests/ds2_perturbed_5deg.star"
        DIR  = "/home/rsanchez/cryo/data/EMPIAR-download/EMPIAR-10166/data"

        # align_star has no n_first_particles param — write a 5-particle star file
        star_5 = tempfile.mktemp(suffix=".star")
        star_data = starfile.read(STAR)
        starfile.write({"optics": star_data["optics"],
                        "particles": star_data["particles"].iloc[:5]}, star_5)

        pm = ProjectionMatcher(reference_vol=REF)
        assert pm.use_so3_interpolation

        out_star = tempfile.mktemp(suffix=".star")
        try:
            result = pm.align_star(
                particles=star_5,
                starFnameOut=out_star,
                data_rootdir=DIR,
                batch_size=1,
                device=f"cuda:{gpu_id}",
                n_cpus=0,  # no DataLoader workers — avoids slow spawn reimport of cryoPARES
            )
            n_out = len(result.particles_md)
            assert n_out == 5, f"expected 5 particles out, got {n_out}"
            print(f"  [OK] GPU forward pass completed, {n_out} particles aligned")
        finally:
            for f in (out_star, star_5):
                if os.path.exists(f):
                    os.unlink(f)
    finally:
        main_config.projmatching.use_so3_interpolation = False


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1, help="GPU index for integration test")
    args = parser.parse_args()

    print("\n=== _precompute_so3_interp_neighbors ===")
    test_neighbor_table_shape()
    test_corner_has_three_valid_neighbors()
    test_center_has_all_six_neighbors()
    test_neighbor_indices_correct()
    test_boundary_indices_clamped_to_self()

    print("\n=== _so3_interpolate_euler_winner ===")
    test_interpolation_output_shape()
    test_interpolation_bounded_by_half_step()
    test_boundary_winner_returns_zero_on_invalid_axis()
    test_parabolic_peak_at_winner()
    test_parabolic_peak_offset_from_winner()

    print("\n=== Integration (GPU) ===")
    test_projmatcher_init_with_flag(gpu_id=args.gpu)
    if torch.cuda.is_available():
        test_tiny_gpu_projmatching(gpu_id=args.gpu)
    else:
        print("  [SKIP] no CUDA device available")

    print("\n✓ All tests passed")
