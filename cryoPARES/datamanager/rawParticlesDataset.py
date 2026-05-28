import os.path as osp
import subprocess
import warnings
from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
from starstack.constants import RELION_IMAGE_FNAME
from starstack.particlesStar import ParticlesStarSet, split_particle_and_fname
from torch.utils.data import Dataset
from typing import Optional, List, Tuple

from cryoPARES.configs.datamanager_config.particlesDataset_config import CtfCorrectionType
from cryoPARES.constants import (
    RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_EULER_CONVENTION,
    RELION_ORI_POSE_CONFIDENCE_NAME,
    BATCH_IDS_NAME, BATCH_POSE_NAME, BATCH_ORI_IMAGE_NAME, BATCH_ORI_CTF_NAME,
)
from cryoPARES.datamanager.ctf.rfft_ctf import correct_ctf

warnings.filterwarnings("ignore", "Gimbal lock detected. Setting third angle to zero since it "
                                  "is not possible to uniquely determine all angles.")


class RawParticlesDataset(Dataset, ABC):
    """Minimal particle dataset: raw images at native resolution + CTF envelope + pose.

    Does NOT resize, normalize, augment, or apply any NN-specific preprocessing.
    Suitable for projection matching and other tasks that operate at native resolution.

    Batch dict keys: BATCH_ORI_IMAGE_NAME (H,W), BATCH_ORI_CTF_NAME (H,W//2+1 or None),
                     BATCH_POSE_NAME (rotMat, xyShiftAngs, confidence), BATCH_IDS_NAME.
    """

    def __init__(
            self,
            halfset: Optional[int] = None,
            ctf_correction: str = "phase_flip",
            subset_idxs: Optional[List[int]] = None,
            require_angles: bool = False,
            min_maxProb: Optional[float] = None,
    ):
        super().__init__()
        self.halfset = halfset
        self.subset_idxs = subset_idxs
        self.require_angles = require_angles
        self.min_maxProb = min_maxProb
        self._particles = None

        assert ctf_correction in (item.value for item in CtfCorrectionType), (
            f"Invalid ctf_correction: {ctf_correction}"
        )
        # Strip the "concat_" prefix — raw dataset always returns the original image,
        # so the "concat" variant is meaningless here.  We only need the envelope.
        ctf_correction_base = ctf_correction.removeprefix("concat_")

        if ctf_correction_base == "none":
            self._correctCtf = self._correctCtfNone
        elif ctf_correction_base == "phase_flip":
            self._correctCtf = self._correctCtfPhase
        elif ctf_correction_base == "ctf_multiply":
            raise NotImplementedError("ctf_multiply not implemented")

        # Used by subclasses (e.g. ParticlesDataset._correctCtfPhase)
        self.ctf_correction_do_concat = ctf_correction.startswith("concat")
        self.ctf_correction = ctf_correction_base

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def load_ParticlesStarSet(self) -> ParticlesStarSet:
        raise NotImplementedError()

    # ── Particle loading / filtering ─────────────────────────────────────────

    def _load_ParticlesStarSet(self):
        part_set = self.load_ParticlesStarSet()
        self._particles = part_set
        assert len(part_set) > 0, "Error, no particles were found in the star file"

        if self.require_angles:
            missing = [name for name in RELION_ANGLES_NAMES
                       if name not in part_set.particles_md.columns]
            if missing:
                raise ValueError(
                    f"Star file is missing angle columns required for training: {missing}. "
                    "Training requires pre-aligned particles with poses "
                    "(rlnAngleRot, rlnAngleTilt, rlnAnglePsi)."
                )

        if self.subset_idxs is not None:
            self._particles = self._particles.createSubset(idxs=self.subset_idxs)

        if self.halfset is not None:
            if "rlnRandomSubset" not in self._particles.particles_md:
                half1, half2 = train_test_split(
                    self._particles.particles_md.index,
                    test_size=0.5, random_state=11, shuffle=True
                )
                self._particles.particles_md.loc[:, "rlnRandomSubset"] = 1
                self._particles.particles_md.loc[half2, "rlnRandomSubset"] = 2

            subsetNums = self._particles.particles_md["rlnRandomSubset"].values
            _subsetNums = set(subsetNums)
            assert min(_subsetNums) >= 1 and max(_subsetNums) <= 2
            idxs = np.where(subsetNums == self.halfset)[0]
            self._particles = self._particles.createSubset(idxs=idxs)

        if self.min_maxProb is not None:
            maxprob = self._particles.particles_md.get(RELION_ORI_POSE_CONFIDENCE_NAME)
            if maxprob is not None:
                idxs = np.where(maxprob >= self.min_maxProb)[0]
                self._particles = self._particles.createSubset(idxs=idxs)

        return self._particles

    @property
    def particles(self) -> ParticlesStarSet:
        if self._particles is None:
            self._particles = self._load_ParticlesStarSet()
        return self._particles

    # ── Metadata helpers ──────────────────────────────────────────────────────

    @property
    def sampling_rate(self) -> float:
        """Original particle image sampling rate in Å/pixel."""
        return self.particles.sampling_rate

    def original_sampling_rate(self) -> float:
        return self.particles.sampling_rate

    def original_image_size(self) -> int:
        images_sizes = set(int(x) for x in self.particles.optics_md["rlnImageSize"].values)
        assert len(images_sizes) == 1, (
            "Several rlnImageSize values in starfile; only one is supported"
        )
        return images_sizes.pop()

    # ── CTF helpers ───────────────────────────────────────────────────────────

    def _correctCtfPhase(self, img: torch.Tensor, md_row, optics_data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CTF envelope without modifying the image.

        For RawParticlesDataset the particle image is always returned unchanged;
        subclasses (ParticlesDataset) override this to also apply the phase flip.
        """
        ctf, _ = correct_ctf(
            img,
            float(optics_data["rlnImagePixelSize"].item()),
            dfu=md_row["rlnDefocusU"], dfv=md_row["rlnDefocusV"],
            dfang=md_row["rlnDefocusAngle"],
            volt=float(optics_data["rlnVoltage"].item()),
            cs=float(optics_data["rlnSphericalAberration"].item()),
            w=float(optics_data["rlnAmplitudeContrast"].item()),
            mode="phase_flip", fftshift=True,
        )
        return img, ctf.real

    def _correctCtfNone(self, img: torch.Tensor, md_row, optics_data) -> Tuple[torch.Tensor, None]:
        return img, None

    # ── Per-item loading ──────────────────────────────────────────────────────

    def _getIdx(self, item: int):
        try:
            img_ori, md_row = self.particles[item]
        except ValueError:
            print(f"Error retrieving item {item}")
            raise

        optics_group_num = int(md_row["rlnOpticsGroup"])
        optics_data = self.particles.optics_md.query(f"rlnOpticsGroup == {optics_group_num}")

        iid = md_row[RELION_IMAGE_FNAME]
        img_ori = torch.FloatTensor(img_ori)          # (H, W)

        # Compute CTF envelope; img_ori is NOT modified (stays raw).
        _, ctf_ori = self._correctCtf(img_ori.unsqueeze(0), md_row, optics_data)

        degEuler = torch.FloatTensor([md_row.get(name, 0) for name in RELION_ANGLES_NAMES])
        xyShiftAngs = torch.FloatTensor([md_row.get(name, 0) for name in RELION_SHIFTS_NAMES])
        confidence = torch.FloatTensor([md_row.get(RELION_ORI_POSE_CONFIDENCE_NAME, 1)])

        r = R.from_euler(RELION_EULER_CONVENTION, degEuler.numpy(), degrees=True)
        rotMat = torch.FloatTensor(r.as_matrix())

        return iid, img_ori, (rotMat, xyShiftAngs, confidence), ctf_ori

    def __getitem__(self, item: int):
        iid, img_ori, pose, ctf_ori = self._getIdx(item)
        return {
            BATCH_IDS_NAME: iid,
            BATCH_ORI_IMAGE_NAME: img_ori,
            BATCH_ORI_CTF_NAME: ctf_ori,
            BATCH_POSE_NAME: pose,
        }

    def __len__(self):
        return len(self.particles)


class RawParticlesRelionStarDataset(RawParticlesDataset):
    """RawParticlesDataset backed by a RELION star file."""

    def __init__(
            self,
            particles_star_fname: str,
            particles_dir: Optional[str],
            subset_idxs: Optional[List[int]] = None,
            **kwargs,
    ):
        super().__init__(subset_idxs=subset_idxs, **kwargs)
        self._star_fname = str(particles_star_fname)
        self._datadir = osp.expanduser(particles_dir) if particles_dir is not None else None

    def load_ParticlesStarSet(self) -> ParticlesStarSet:
        ps = ParticlesStarSet(starFname=self._star_fname, particlesDir=self._datadir)

        ulimit = subprocess.run(["ulimit -n"], check=True, capture_output=True, shell=True)
        assert ulimit.returncode == 0, "Error, ulimit -n command failed"
        ulimit = int(ulimit.stdout.decode().strip())
        stackFnames = ps.particles_md["rlnImageName"].map(
            lambda x: split_particle_and_fname(x)["basename"]
        )
        n_stacks = len(stackFnames.unique())
        assert n_stacks < 5 * ulimit, (
            f"Error, the number of particle stacks is too large ({n_stacks}) given current "
            f"number of allowed file handlers {ulimit}. Increase the limit with 'ulimit -n NUMBER'."
        )
        return ps
