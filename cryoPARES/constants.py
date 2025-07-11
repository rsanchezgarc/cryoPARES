from typing import List

import torch
from starstack.constants import RELION_EULER_CONVENTION, RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, \
    RELION_PRED_POSE_CONFIDENCE_NAME, RELION_ORI_POSE_CONFIDENCE_NAME, RELION_IMAGE_FNAME

PROJECT_NAME: str = "cryoPARES"
SCRIPT_ENTRY_POINT: str = PROJECT_NAME + "__ENTRY_POINT"
float32_matmul_precision: str = "high"

BATCH_IDS_NAME: str = "idd"
BATCH_PARTICLES_NAME: str = "particle"
BATCH_POSE_NAME: str = "pose"
BATCH_MD_NAME: str = "md"
BATCH_ORI_IMAGE_NAME: str = "oriparticle"
BATCH_ORI_CTF_NAME: str = "oriCTF"

DIRECTIONAL_ZSCORE_NAME = "rlnDirectinalZscore"
PROJECTION_MATCHING_SCORE = "rlnProjMatchScore"

#Name for directories or files
DATA_SPLITS_BASENAME: str = "data_splits"
TRAINING_DONE_TEMPLATE: str = "DONE_TRAINING.txt" #f"DONE-pid_%(pid)d.txt"
BEST_CHECKPOINT_BASENAME: str = "best.ckpt"
BEST_MODEL_SCRIPT_BASENAME: str = "best_script.pt"
BEST_DIRECTIONAL_NORMALIZER: str = "best_directional_normalizer.pt"

DEFAULT_DTYPE_PROJMATCHING: torch.dtype = torch.float32