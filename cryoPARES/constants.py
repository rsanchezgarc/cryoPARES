from typing import List
from starstack.constants import RELION_EULER_CONVENTION, RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, \
    RELION_PRED_POSE_CONFIDENCE_NAME, RELION_ORI_POSE_CONFIDENCE_NAME, RELION_IMAGE_FNAME

PROJECT_NAME: str = "cryoPARES"
float32_matmul_precision: str = "high"

BATCH_IDS_NAME: str = "idd"
BATCH_PARTICLES_NAME: str = "particle"
BATCH_POSE_NAME: str = "pose"
BATCH_MD_NAME: str = "md"

#Name for directories or files
DATA_SPLITS_BASENAME= "data_splits"
TRAINING_DONE_TEMPLATE = "DONE_TRAINING.txt" #f"DONE-pid_%(pid)d.txt"
BEST_CHECKPOINT_BASENAME="best.ckpt"
BEST_MODEL_SCRIPT_BASENAME= "best_script.pt"