"""
Generate a RELION-style STAR file with random SO(3) poses and CTF parameters.

Used by simulateParticles.py when --n_particles is given instead of a full
particle STAR: the caller provides an optics-only STAR (or any STAR whose
optics table carries rlnImagePixelSize, rlnImageSize, rlnVoltage, …) and
this module synthesises the particles block with random orientations and CTF.
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation


def generate_random_particles_star(
    optics_star: str,
    n_particles: int,
    output_dir: str,
    basename: str,
    defocus_min: float = 5000.0,
    defocus_max: float = 25000.0,
    astigmatism_std: float = 200.0,
    shift_range_A: float = 0.0,
    random_seed: Optional[int] = None,
) -> Tuple[str, pd.DataFrame]:
    """
    Generate a RELION STAR file with ``n_particles`` random particles.

    Orientations are sampled uniformly over SO(3) using scipy's Rotation.random.
    Defocus values are drawn uniformly in [defocus_min, defocus_max].
    Astigmatism is modelled as DefocusV = DefocusU + N(0, astigmatism_std).
    DefocusAngle is uniform in [0, 180).
    Shifts are uniform in [-shift_range_A, +shift_range_A] (0 = no shifts).

    Args:
        optics_star: Path to a STAR file whose *optics* table will be reused
                     verbatim (must contain rlnImagePixelSize, rlnImageSize,
                     rlnVoltage, rlnSphericalAberration, rlnAmplitudeContrast).
        n_particles:   Number of particles to generate.
        output_dir:    Directory where the generated STAR file is written.
        basename:      Base name; output file is ``<basename>_random.star``.
        defocus_min:   Minimum DefocusU in Ångströms.
        defocus_max:   Maximum DefocusU in Ångströms.
        astigmatism_std: Std of DefocusV offset from DefocusU in Ångströms.
        shift_range_A: Half-range for rlnOriginX/YAngst (0 → no shifts).
        random_seed:   Optional seed for reproducibility.

    Returns:
        Tuple of (path_to_star, optics_df).
    """
    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------ optics
    raw = starfile.read(optics_star)
    if isinstance(raw, dict):
        optics_df = raw.get("optics", raw.get("data_optics", None))
        if optics_df is None:
            # Fall back: first table that looks like an optics block
            for v in raw.values():
                if isinstance(v, pd.DataFrame) and "rlnImagePixelSize" in v.columns:
                    optics_df = v
                    break
        if optics_df is None:
            raise ValueError(
                f"Could not find optics table in {optics_star}. "
                "Expected a dict with key 'optics' or 'data_optics'."
            )
    elif isinstance(raw, pd.DataFrame):
        # Single-block STAR — treat it as optics
        optics_df = raw
    else:
        raise ValueError(f"Unexpected starfile.read result type: {type(raw)}")

    optics_df = optics_df.copy()

    # --------------------------------------------------------- random SO(3) poses
    # scipy Rotation.random gives a proper uniform sample over SO(3).
    # RELION uses intrinsic ZYZ Euler angles: (rot, tilt, psi).
    scipy_seed = int(rng.integers(0, 2**31)) if random_seed is None else random_seed
    rotations = Rotation.random(n_particles, random_state=scipy_seed)
    # 'ZYZ' intrinsic = RELION convention
    eulers = rotations.as_euler("ZYZ", degrees=True)  # (N, 3): rot, tilt, psi
    rot   = eulers[:, 0]
    tilt  = eulers[:, 1]
    psi   = eulers[:, 2]

    # -------------------------------------------------------------------- CTF
    # Sample mean defocus and astigmatism independently, then assign U/V so
    # that DefocusU >= DefocusV as required by the RELION convention.
    # DefocusAngle is the orientation of the astigmatic ellipse — physically
    # independent of the defocus magnitudes, so uniform [0, 180) is correct.
    df_mean = rng.uniform(defocus_min, defocus_max, n_particles)
    df_astig = np.abs(rng.normal(0.0, astigmatism_std, n_particles))
    dfu = np.clip(df_mean + df_astig / 2.0, defocus_min * 0.5, defocus_max * 2.0)
    dfv = np.clip(df_mean - df_astig / 2.0, defocus_min * 0.5, defocus_max * 2.0)
    dfa = rng.uniform(0.0, 180.0, n_particles)

    # ------------------------------------------------------------------ shifts
    if shift_range_A > 0.0:
        sx = rng.uniform(-shift_range_A, shift_range_A, n_particles)
        sy = rng.uniform(-shift_range_A, shift_range_A, n_particles)
    else:
        sx = np.zeros(n_particles, dtype=np.float32)
        sy = np.zeros(n_particles, dtype=np.float32)

    # --------------------------------------------------------- particles table
    optics_group = int(optics_df["rlnOpticsGroup"].iloc[0])
    particles_df = pd.DataFrame({
        "rlnImageName":    [f"{i+1}@dummy.mrcs" for i in range(n_particles)],
        "rlnOpticsGroup":  [optics_group] * n_particles,
        "rlnAngleRot":     rot.astype(np.float32),
        "rlnAngleTilt":    tilt.astype(np.float32),
        "rlnAnglePsi":     psi.astype(np.float32),
        "rlnOriginXAngst": sx.astype(np.float32),
        "rlnOriginYAngst": sy.astype(np.float32),
        "rlnDefocusU":     dfu.astype(np.float32),
        "rlnDefocusV":     dfv.astype(np.float32),
        "rlnDefocusAngle": dfa.astype(np.float32),
        "rlnCtfBfactor":   np.zeros(n_particles, dtype=np.float32),
        "rlnCoordinateX":  np.zeros(n_particles, dtype=np.float32),
        "rlnCoordinateY":  np.zeros(n_particles, dtype=np.float32),
    })

    # ------------------------------------------------------------- write STAR
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{basename}_random.star")
    starfile.write({"optics": optics_df, "particles": particles_df}, out_path, overwrite=True)

    return out_path, optics_df