"""
Half-model consensus pruning for ``allCombinations`` inference.

When inference is run with ``model_halfset="allCombinations"``, every particle is
predicted twice: once by the half1 model and once by the half2 model. A particle whose
two independent predictions agree is trustworthy; one where the two models disagree is
likely a bad/ambiguous particle.

This module discovers the per-combination prediction star files written by the inference
pipeline and, for each data half, compares the *matching*-half prediction with the
*cross*-half prediction, dropping particles whose symmetry-aware angular disagreement
exceeds a threshold. The surviving particles keep the **cross-half** (non-overfit) pose.

The heavy lifting (matching particles, computing symmetry-aware angular errors, dropping)
is reused from :func:`cryoPARES.scripts.consensus_alignment.consensus_alignment` with
``consensus_mode="drop"`` (which retains the first input file's poses) and exact matching
on ``rlnImageName``.
"""
import glob
import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd
import starfile

from cryoPARES.scripts.consensus_alignment import consensus_alignment


# Per-combination prediction star file naming, shared with the inference entrypoints so
# the 4 allCombinations passes write distinct (non-colliding) files.
def allcombinations_star_suffix(data_halfset: str, model_halfset: str, extension: str = "star") -> str:
    """Filename suffix that disambiguates an allCombinations (data, model) pass."""
    return f"_data_{data_halfset}_model_{model_halfset}.{extension}"


_COMBO_RE = re.compile(r"^(?P<base>.+)_data_(?P<data>half[12])_model_(?P<model>half[12])\.star$")


def _discover_combination_files(results_dir: str) -> Dict[Tuple[str, str, str], str]:
    """Map (basename, data_half, model_half) -> star path for all 4 allCombinations passes."""
    combos: Dict[Tuple[str, str, str], str] = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "*_data_half*_model_half*.star"))):
        m = _COMBO_RE.match(os.path.basename(f))
        if m:
            combos[(m["base"], m["data"], m["model"])] = f
    return combos


def _merge_consensus_halves(consensus_half_files: Dict[str, str], out_fname: str) -> None:
    """Concatenate the per-half consensus star files into a single merged star, tagging
    each particle with rlnRandomSubset (1/2) from its data half."""
    all_particles = []
    optics_df = None
    for data_half, sf in sorted(consensus_half_files.items()):
        data = starfile.read(sf)
        if isinstance(data, dict):
            particles = data.get("particles")
            if optics_df is None:
                optics_df = data.get("optics")
        else:
            particles = data
        if not isinstance(particles, pd.DataFrame):
            continue
        particles = particles.copy()
        particles["rlnRandomSubset"] = 1 if data_half == "half1" else 2
        all_particles.append(particles)
    if not all_particles:
        return
    merged = pd.concat(all_particles, axis=0, ignore_index=True)
    payload = {"particles": merged}
    if optics_df is not None:
        payload["optics"] = optics_df
    starfile.write(payload, out_fname, overwrite=True)
    print(f"Merged consensus output saved: {out_fname}")


def run_consensus_phase(
    results_dir: str,
    symmetry: str,
    thr_degs: float,
    particles_dir: Optional[str] = None,
    reconstruct: bool = True,
    n_jobs: int = 1,
    use_cuda: bool = True,
    reference_mask: Optional[str] = None,
) -> Dict[str, str]:
    """
    Run half-model consensus pruning over the per-combination prediction star files in
    ``results_dir`` (written by the allCombinations passes).

    For each data half, compares the matching-half and cross-half predictions, drops
    particles whose disagreement exceeds ``thr_degs``, and writes a consensus star file
    keeping the cross-half pose (plus a per-particle ``rlnPoseConsensusAngularError``
    column). Optionally reconstructs each half from the pruned set and reports FSC.

    Pure I/O over ``results_dir`` so it is reusable for any ``n_jobs``.

    :return: mapping ``{"half1": path, "half2": path}`` of the per-half consensus star files.
    """
    combos = _discover_combination_files(results_dir)
    if not combos:
        raise FileNotFoundError(
            f"No per-combination prediction star files (*_data_halfX_model_halfY.star) found in "
            f"{results_dir}. Consensus pruning requires model_halfset='allCombinations'."
        )

    bases = sorted({key[0] for key in combos})
    if len(bases) > 1:
        warnings.warn(
            f"Found multiple input basenames in {results_dir} ({bases}); running consensus "
            f"independently per basename. Reconstruction/FSC uses the last basename only."
        )

    consensus_half_files: Dict[str, str] = {}
    for base in bases:
        per_base_files: Dict[str, str] = {}
        for data_half in ("half1", "half2"):
            other = "half2" if data_half == "half1" else "half1"
            matching = combos.get((base, data_half, data_half))
            cross = combos.get((base, data_half, other))
            if matching is None or cross is None:
                warnings.warn(
                    f"Skipping consensus for data {data_half} (basename '{base}'): missing "
                    f"{'matching' if matching is None else 'cross'}-half prediction file."
                )
                continue
            out_fname = os.path.join(results_dir, f"{base}_consensus_{data_half}.star")
            print(f"\n=== Consensus pruning for data {data_half} "
                  f"(cross-half pose from model {other}, threshold {thr_degs}°) ===")
            # cross file first -> consensus_mode="drop" keeps its (cross-half) poses
            consensus_alignment(
                input_stars=[cross, matching],
                output_star=out_fname,
                symmetry=symmetry,
                consensus_mode="drop",
                angular_threshold_degs=thr_degs,
                match_keys=["rlnImageName"],
            )
            per_base_files[data_half] = out_fname

        # Merge halves for this basename
        if per_base_files:
            merged_fname = os.path.join(results_dir, f"{base}_consensus.star")
            _merge_consensus_halves(per_base_files, merged_fname)
        consensus_half_files = per_base_files  # last basename wins for reconstruction

    if reconstruct and len(consensus_half_files) == 2:
        _reconstruct_and_fsc(consensus_half_files, results_dir, symmetry, particles_dir,
                             n_jobs=n_jobs, use_cuda=use_cuda, reference_mask=reference_mask)

    return consensus_half_files


def _reconstruct_and_fsc(consensus_half_files: Dict[str, str], results_dir: str, symmetry: str,
                         particles_dir: Optional[str], n_jobs: int, use_cuda: bool,
                         reference_mask: Optional[str]) -> None:
    """Reconstruct each consensus half and report gold-standard FSC."""
    from cryoPARES.reconstruction.reconstruct import reconstruct_starfile
    from cryoPARES.scripts.computeFsc import compute_fsc
    from cryoPARES.utils.reconstructionUtils import get_vol

    half_maps: Dict[str, str] = {}
    for data_half, star_fname in sorted(consensus_half_files.items()):
        out_map = os.path.join(results_dir, f"reconstruction_consensus_{data_half}.mrc")
        print(f"\nReconstructing consensus {data_half} from {os.path.basename(star_fname)} ...")
        reconstruct_starfile(
            particles_star_fname=star_fname,
            symmetry=symmetry,
            output_fname=out_map,
            particles_dir=particles_dir,
            n_jobs=n_jobs,
            use_cuda=use_cuda,
        )
        half_maps[data_half] = out_map

    vol1, sampling_rate = get_vol(half_maps["half1"], pixel_size=None)
    vol2, _ = get_vol(half_maps["half2"], pixel_size=None)
    mask = get_vol(reference_mask, pixel_size=None)[0] if reference_mask is not None else None
    print("\nComputing FSC between consensus half-maps...")
    fsc, spatial_freq, resolution_A, (res_05, res_0143) = compute_fsc(
        vol1.cpu().numpy(), vol2.cpu().numpy(), sampling_rate, mask=mask
    )
    print(f"Consensus resolution at FSC=0.143 ('gold-standard'): {res_0143:.3f} Å")
    print(f"Consensus resolution at FSC=0.5: {res_05:.3f} Å")
